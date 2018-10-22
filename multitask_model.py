import math
from collections import OrderedDict
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from model import supported_rnns, supported_rnns_inv
from model import SequenceWise, MaskConv, InferenceBatchSoftmax, BatchRNN, Lookahead, DeepSpeech


class MtAccent(DeepSpeech):

    @overrides
    def __init__(self, accents_size, bottleneck_size=40, rnn_type=nn.LSTM, labels="abc", 
                rnn_hidden_size=768, nb_layers=5, 
                audio_conf=None, bidirectional=True, context=20,
                side_nb_layers=4, side_rnn_hidden_size=768,
                side_rnn_type=nn.LSTM, nb_shared_layers=2):

        assert nb_shared_layers <= nb_layers,'There must be less shared layers than main layers (nb_shared_layers <= nb_layers).'
        assert nb_shared_layers <= side_nb_layers,'There must be less shared layers than side layers (nb_shared_layers <= side_nb_layers).'
        super(MtAccent, self).__init__(rnn_type=rnn_type, labels=labels, 
                                        rnn_hidden_size=rnn_hidden_size, nb_layers=nb_shared_layers, 
                                        audio_conf=audio_conf, bidirectional=bidirectional, 
                                        context=context)
        # add extended rnn main layers
        add_rnns = []
        for i in range(nb_shared_layers, nb_layers):
            rnn = BatchRNN(input_size=rnn_hidden_size + bottleneck_size, 
                        hidden_size=rnn_hidden_size + bottleneck_size,
                        rnn_type=rnn_type, 
                        bidirectional=bidirectional, 
                        batch_norm=False)
            add_rnns.append((f'i', rnn))
        self.add_rnns = nn.Sequential(OrderedDict(add_rnns))

        # Shared
        self.nb_shared_layers=nb_shared_layers

        # Side RNNs
        side_rnns = []
        rnn = BatchRNN(input_size=rnn_hidden_size, 
                        hidden_size=side_rnn_hidden_size,
                        rnn_type=side_rnn_type, 
                        bidirectional=bidirectional, 
                        batch_norm=False)
        side_rnns.append(('side_0', rnn))

        for i in range(1, side_nb_layers):
            rnn = BatchRNN(input_size=side_rnn_hidden_size, 
                            hidden_size=side_rnn_hidden_size,
                            rnn_type=side_rnn_type, 
                            bidirectional=bidirectional)
            side_rnns.append((f'side_{i}', rnn))

        self.side_rnns = nn.Sequential(OrderedDict(side_rnns))
        
        # soft max and bottleneck        
        self.funnel = nn.Linear(side_rnn_hidden_size, bottleneck_size)

        self.side_fc = nn.Linear(bottleneck_size, accents_size, bias=False)
        self.side_softmax = nn.LogSoftmax(dim=1)
        
        # fc 
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size + bottleneck_size),
            nn.Linear(rnn_hidden_size + bottleneck_size, len(self._labels), bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        # params to save
        self._accents_size = accents_size
        self._bottleneck_size = bottleneck_size
        self._side_nb_layers = side_nb_layers
        self._side_rnn_hidden_size = side_rnn_hidden_size
        self._side_rnn_type = side_rnn_type
        self._nb_shared_layers = nb_shared_layers


    @overrides
    def forward(self, x, lenghts):
        #print('NEW FORWARD')
        #print('X base', x.size())
        lenghts = lenghts.cpu().int()
        output_lenghts = self.get_seq_lens(lenghts)
        x, _ = self.conv(x, output_lenghts)
        #print('X conv', x.size())
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        #print('X view', x.size())
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        #print('X transpose', x.size())
        # Initialize shared layers
        for i in range(len(self.rnns)):
            x = self.rnns[i](x, output_lenghts)
        #print('X shared rnn', x.size())


        # Rest of side layers
        side_x = self.side_rnns[0](x, output_lenghts)
        #print('SIDE X rnns', side_x.size())
        for i in range(1, len(self.side_rnns)):
            side_x = self.side_rnns[i](x, output_lenghts)
        #print('SIDE X other rnns', side_x.size())
        bottleneck = self.funnel(side_x)
        #print('SIDE X BOTTLE', bottleneck.size())
        side_x = self.side_fc(bottleneck)
        #print('SIDE X fully co', side_x.size())
        side_x = side_x[-1]
        #print('SIDE X take last', side_x.size())
        side_x = self.side_softmax(side_x)
        #print('SIDE X softmax', side_x.size())

        # Rest of main layers
        concat = torch.cat((x, bottleneck), dim=2)
        #print('CONCAT', concat.size())
        x = self.add_rnns[0](concat, output_lenghts)
        #print('X more rnn', x.size())
        for i in range(1, len(self.add_rnns)):
            x = self.rnns[i](x, output_lenghts)
        #print('X more more rnn', x.size())
        if not self._bidirectional:
            x = self.lookahead(x)
        x = self.fc(x)
        #print('X fully co', x.size())
        x = x.transpose(0, 1)
        #print('X transpose', x.size())
        x = self.inference_softmax(x)
        #print('X infer softmax', x.size())

        return x, output_lenghts, side_x


    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(accents_size=package['accents_size'], rnn_hidden_size=package['hidden_size'], 
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        for x in model.side_rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(accents_size=package['accents_size'], bottleneck_size=package['bottleneck_size'], rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True),
                    side_nb_layers=package['side_nb_layers'], 
                    side_rnn_hidden_size=package['side_rnn_hidden_size'],
                    side_rnn_type=package['side_rnn_type'], 
                    nb_shared_layers=package['nb_shared_layers'])
        model.load_state_dict(package['state_dict'])
        return model