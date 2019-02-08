import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from models.modules import MaskConv, SequenceWise, BatchRNN, InferenceBatchSoftmax, Lookahead, \
                    supported_rnns, supported_rnns_inv

DEBUG = True 

class DeepSpeech(nn.Module):
    def __init__(self, 
                rnn_type=nn.LSTM, 
                labels="abc", 
                rnn_hidden_size=768, 
                nb_layers=5, 
                audio_conf=None,
                bidirectional=True, 
                context=20,
                DEBUG=DEBUG):

        super(DeepSpeech, self).__init__()
        self._DEBUG = DEBUG
        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._nb_layers = nb_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x, lengths):
        if self._DEBUG:
            print('after input ', x.size())
        
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)
        if self._DEBUG:
            print('after conv', x.size())

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        if self._DEBUG:
            print('after view', x.size())
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        if self._DEBUG:
            print('after transpose', x.size())

        for rnn in self.rnns:
            x = rnn(x, output_lengths)
        if self._DEBUG:
            print('after rnn', x.size())

        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        if self._DEBUG:
            print('after fc', x.size())
        x = x.transpose(0, 1)
        if self._DEBUG:
            print('after transpose', x.size())
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        if self._DEBUG:
            print('after softmax', x.size())
        self._DEBUG = False
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['nb_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['nb_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  main_loss_results=None, side_loss_results=None,
                  cer_results=None, wer_results=None, mca_results=None, avg_loss=None, meta=None):
        model = model.module if DeepSpeech.is_parallel(model) else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'nb_layers': model._nb_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'bidirectional': model._bidirectional
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['main_loss_results'] = main_loss_results
            package['side_loss_results'] = side_loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
            package['mca_results'] = mca_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        return model.module._labels if model.is_parallel(model) else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        return model.module._audio_conf if DeepSpeech.is_parallel(model) else model._audio_conf

    @staticmethod
    def get_meta(model):
        m = model.module if DeepSpeech.is_parallel(model) else model
        meta = {
            "version": m._version,
            "hidden_size": m._hidden_size,
            "nb_layers": m._nb_layers,
            "rnn_type": supported_rnns_inv[m._rnn_type]
        }
        return meta

    @staticmethod
    def is_parallel(model):
        return isinstance(model, torch.nn.parallel.DataParallel) or \
               isinstance(model, torch.nn.parallel.DistributedDataParallel)


# Main
if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model._version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model._rnn_type.__name__.lower())
    print("  RNN Layers:       ", model._nb_layers)
    print("  RNN Size:         ", model._hidden_size)
    print("  Classes:          ", len(model._labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model._labels)
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get("window_stride", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))

    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)
