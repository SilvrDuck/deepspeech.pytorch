for RUN in range(5):
    print('#### START 03 - {RUN} ####')
    DEV = False
    EPOCHS = 30

    DEBUG = False

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from warpctc_pytorch import CTCLoss
    #torch.multiprocessing.set_start_method("spawn")

    import parameters

    # Allows to load modules from parent directory
    import time
    import inspect, sys, os, json
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(inspect.getfile(inspect.currentframe())))))

    from pathlib import Path
    from os import makedirs
    from collections import OrderedDict

    from tqdm import tqdm

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from data.data_loader import create_binarizer, get_accents_counts
    from utils import count_parameters
    from models.modules import MaskConv, SequenceWise, BatchRNN, InferenceBatchSoftmax, Lookahead, \
                        supported_rnns, supported_rnns_inv

    from tensorboardX import SummaryWriter

    import math

    from multitask_loss import MtLoss
    from torch.utils.data import DataLoader, Dataset
    from decoder import GreedyDecoder, BeamCTCDecoder

    param = parameters.get_parameters(dev=DEV, epochs=EPOCHS, us_en=False)


    #### UTILITIES

    def val_cnts(list_):
        return pd.Series(list_).value_counts()

    def extract_num (s):
        return ''.join([c if c.isdigit() else '' for c in s])

    def ids_list(manifest):
        ids = []
        with open(manifest) as f:
            for l in f:
                s = l.split('/')
                ids.append(f'{s[3]}-{s[5].split(".")[0]}')
        return ids

    def make_accent_dict(manifest_path):
        accent_dict = {}
        class_dict = {}
        with open(manifest_path) as f:
            for l in f:
                wav, txt, acc = l.split(',')
                num = extract_num(wav)
                accent = acc.strip()
                if accent not in class_dict:
                    new_key = 0 if (len(class_dict) == 0) else max(class_dict.values()) + 1
                    class_dict[accent] = new_key
                accent_dict[num] = class_dict[accent]
        return accent_dict, {v: k for k, v in class_dict.items()}

    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    ##### DATA


    class KaldiDeepspeechDataset(Dataset):
        """Defines an iterator over the dataset. This class is intended to be used with PyTorch DataLoader"""

        def __init__(self, data_path, labels, sample_ids, transcripts_path,
                     accent_id_dict, ivectors_path=None):

            self.data_path = data_path
            self.ivectors_path = ivectors_path
            self.transcripts_path = transcripts_path
            self.accent_id_dict = accent_id_dict
            self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
            if isinstance(sample_ids, list):
                self._datafiles = sample_ids
            else:
                with open(sample_ids) as f:
                    self._datafiles = [x.strip() for x in f.readlines()]

        def __getitem__(self, index):
            file_idx = self._datafiles[index]
            with open(os.path.join(self.data_path, file_idx)) as f:
                sample = json.load(f)
            sample = torch.FloatTensor(sample)

            target = self.accent_id_dict[extract_num(self._datafiles[index])]

            s_id = file_idx.split('-')[-1]

            transcript_path = f'{self.transcripts_path}sample-{s_id}.txt'
            transcript = self.parse_transcript(transcript_path)             

            if self.ivectors_path is None:
                return torch.FloatTensor(sample), target, transcript
            else:
                with open(os.path.join(self.ivectors_path, self._datafiles[index])) as f:
                    ivect = json.load(f)
                return torch.FloatTensor(sample), target, transcript, torch.FloatTensor(ivect)

        def parse_transcript(self, transcript_path):
            with open(transcript_path, 'r', encoding='utf8') as transcript_file:
                transcript = transcript_file.read().replace('\n', '')
            transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
            return transcript

        def __len__(self):

            return len(self._datafiles)

    def collate_fn(batch_tot):
        """This function takes list of samples and assembles a batch. It is intended to used in PyTorch DataLoader."""
        batch = list(zip(*batch_tot))
        ivect = None

        if len(batch) == 3:
            input_, acc, trs = batch
        elif len(batch) == 4:
            input_, acc, trs, ivect = batch

        input_lens = torch.tensor([len(r) for r in input_])
        acc = torch.tensor(acc)

        input_ = nn.utils.rnn.pad_sequence(input_, batch_first=True)

        target_lens = torch.tensor([len(t) for t in trs])

        if ivect is not None:
            ivect = nn.utils.rnn.pad_sequence(ivect, batch_first=True)
            ivect = tile(ivect, 1, 10)
            ivect = ivect[:, :input_.size(1), :]
            input_ = torch.cat([input_, ivect], dim=2)

        __, idx = input_lens.sort(descending=True)

        targets = np.array(trs)[idx]
        targets = torch.tensor([t for target in targets for t in target])

        return input_[idx], input_lens[idx].int(), targets.int(), target_lens[idx].int(), acc[idx].int()

    class KaldiDeepspeechDataLoader(DataLoader):
        def __init__(self, *args, **kwargs):
            """
            Creates a data loader for SpeechDatasets.
            """
            super(KaldiDeepspeechDataLoader, self).__init__(*args, **kwargs)
            self.collate_fn = collate_fn


    accent_id_dict, accent_dict = make_accent_dict(param['train_manifest'])

    train_dataset = KaldiDeepspeechDataset(data_path=param['train_kaldi'],
                                  labels=param['labels'],
                                  sample_ids=ids_list(param['train_manifest']), 
                                  transcripts_path=param['train_transcripts'],
                                  accent_id_dict=accent_id_dict,
                                  ivectors_path=None)

    train_loader = KaldiDeepspeechDataLoader(train_dataset, 
                                    shuffle=True, 
                                    num_workers=0,#param['num_worker'],
                                    batch_size=param['batch_size'])

    test_dict, __ = make_accent_dict(param['test_manifest'])

    test_dataset = KaldiDeepspeechDataset(data_path=param['test_kaldi'],
                                  labels=param['labels'],
                                  sample_ids=ids_list(param['test_manifest']), 
                                  transcripts_path=param['test_transcripts'],
                                  accent_id_dict=test_dict,
                                  ivectors_path=None)

    test_loader = KaldiDeepspeechDataLoader(test_dataset, 
                                    shuffle=True, 
                                    num_workers=param['num_worker'],
                                    batch_size=param['batch_size'])

    #### MODEL

    class Head(nn.Module):
        def __init__(self, 
                    rnn_type=nn.LSTM, 
                    labels="abc", 
                    rnn_hidden_size=768, 
                    nb_layers=5, 
                    bidirectional=True,
                    DEBUG=False):

            super(Head, self).__init__()

            # model metadata needed for serialization/deserialization
            self._DEBUG = DEBUG
            self._hidden_size = rnn_hidden_size
            self._nb_layers = nb_layers
            self._rnn_type = rnn_type
            self._labels = labels
            self._bidirectional = bidirectional

            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            ))

            rnn_input_size = 320

            rnns = []
            rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, batch_norm=False)
            rnns.append(('0', rnn))
            for x in range(nb_layers - 1):
                rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                               bidirectional=bidirectional)
                rnns.append(('%d' % (x + 1), rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))


        def forward(self, x, lengths):
            if self._DEBUG:
                print('')
                print('# BEGIN HEAD #')
                print('input', x.size())

            lengths = lengths.cpu().int()
            output_lengths = self.get_seq_lens(lengths)

            x = x.view(x.size(0), 1, x.size(1), x.size(2))
            x = x.transpose(2, 3)
            if self._DEBUG:
                print('after view transpose', x.size())

            x, _ = self.conv(x, output_lengths)
            if self._DEBUG:
                print('after conv', x.size())

            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
            if self._DEBUG:
                print('after view transpose', x.size())

            for rnn in self.rnns:
                x = rnn(x, output_lengths)
            if self._DEBUG:
                print('after rnn', x.size())

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

    class FC(nn.Module):
        def __init__(self, 
                     rnn_hidden_size=768,
                     labels='abc',
                     DEBUG=False,):

            super(FC, self).__init__()

            self._DEBUG = DEBUG
            self._labels = labels
            num_classes = len(self._labels)

            fully_connected = nn.Sequential(
                nn.BatchNorm1d(rnn_hidden_size),
                nn.Linear(rnn_hidden_size, num_classes, bias=False)
            )
            self.fc = nn.Sequential(
                SequenceWise(fully_connected),
            )
            self.inference_softmax = InferenceBatchSoftmax()


        def forward(self, x):
            if self._DEBUG:
                print('')
                print('# BEGIN FC #')
                print('input', x.size())
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

            x = x.transpose(0, 1)
            if self._DEBUG:
                print('after transpose', x.size())

            self._DEBUG = False
            return x


    class AccentClassifier(nn.Module):
        def __init__(self,
                     input_size,
                     accents={},
                     DEBUG=False,):

            super(AccentClassifier, self).__init__()

            num_classes = len(accents)
            self._DEBUG = DEBUG

            self.fc = nn.Sequential(
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes),
                nn.ReLU(),
            )

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):     
            if self._DEBUG:
                print('')
                print('# BEGIN Acc #')
                print('input', x.size())

            x = x.mean(dim=0)

            if self._DEBUG:
                print('after mean', x.size())

            x = self.fc(x)

            if self._DEBUG:
                print('after fc', x.size())

            x = self.softmax(x)

            if self._DEBUG:
                print('after softmax', x.size())

            self._DEBUG = False
            return x

    class Network(nn.Module):
        def __init__(self, 
                    rnn_type=nn.LSTM, 
                    labels="abc", 
                    accents={},
                    rnn_hidden_size=768, 
                    nb_layers=5, 
                    audio_conf=None,
                    bidirectional=True,
                    DEBUG=False):

            super(Network, self).__init__()

            self.Head = Head(rnn_type=rnn_type, 
                            labels=labels, 
                            rnn_hidden_size=rnn_hidden_size, 
                            nb_layers=nb_layers, 
                            bidirectional=bidirectional,
                            DEBUG=DEBUG)

            self.FC = FC(rnn_hidden_size=rnn_hidden_size,
                         labels=labels,
                         DEBUG=DEBUG,)

            self.Acc = AccentClassifier(input_size=800,
                                      accents=accents,
                                      DEBUG=DEBUG)

        def forward(self, x, lengths):
            x, out_len = self.Head(x, lengths)
            x_fc = self.FC(x)
            x_acc = self.Acc(x)
            return x_fc, x_acc, out_len

    ### DEC


    crit_fc, crit_acc = CTCLoss(), nn.CrossEntropyLoss()
    decoder = BeamCTCDecoder(param['labels'], lm_path=param['lm_path'],
                            alpha=0.8, beta=1.,
                            cutoff_top_n=40, cutoff_prob=1.0,
                            beam_width=100, num_processes=param['num_worker'])
    target_decoder = GreedyDecoder(param['labels'])


    #### TRAIN


    def check_wer(targets, targets_len, out, output_len):
        split_targets = []
        offset = 0
        for size in targets_len:
            split_targets.append(targets[offset:offset + size])
            offset += size

        decoded_output, _ = decoder.decode(out.data.transpose(0,1), output_len)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if False:
            print('targets', targets)
            print('split_targets', split_targets)
            print('out', out)
            print('output_len', output_len)
            print('decoded', decoded_output)
            print('target', target_strings)

        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer += decoder.wer(transcript, reference) / float(len(reference.split()))
            #cer += decoder.cer(transcript, reference) / float(len(reference))
        wer /= len(target_strings)
        return wer * 100



    def check_acc(targets, out):
        out_arg = np.argmax(out, axis=1)
        diff = torch.eq(out_arg.int(), targets)
        acc = torch.sum(diff).float()
        return acc / len(targets) * 100

    def get_loss(inputs, inputs_len, targets, targets_len, target_accents, mix):

        inputs = inputs.cuda()
        inputs_len = inputs_len.cuda()
        targets = targets.cuda()
        targets_len = targets_len.cuda()

        # Forward pass
        out_fc, out_acc, output_len = model(inputs, inputs_len)

        out_fc = out_fc.cpu()
        targets = targets.cpu()
        targets_len = targets_len.cpu()
        out_acc = out_acc.cpu()

        if False:
            print('## Outputs train')
            print('out', out.size())
            print('targets', targets.size())
            print('output_len', output_len.size())
            print('targets_len', targets_len.size())

        loss_fc = crit_fc(out_fc, targets, output_len, targets_len)
        loss_acc = crit_acc(out_acc, target_accents.long()) * 3000
        loss = mix * loss_fc + (1 - mix) * loss_acc

        total_loss = (loss, loss_fc, loss_acc)
        return total_loss, out_fc, out_acc, output_len

    def train(epochs, 
              model, 
              train_loader, 
              test_loader, 
              optimizer, 
              silent=True,
              mix=0.5,
              exp_name='__tmp__'):

        # Tensorboard
        tb_path = Path(param['tensorboard_dir']) / exp_name
        makedirs(tb_path, exist_ok=True)
        tb_writer = SummaryWriter(tb_path)

        prev_epoch_val_loss = math.inf
        prev_epoch_wer = math.inf

        ## Train
        for epoch in range(1, param['epochs'] + 1):
            import gc; gc.collect()
            print('')
            print(f'## EPOCH {epoch} ##')
            print(f'Training:')
            model.train()

            # train
            epoch_losses = []
            epoch_losses_fc = []
            epoch_losses_acc = []
            for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                total_loss, __, __, __ = get_loss(*data, mix=mix)
                loss, loss_fc, loss_acc = total_loss

                epoch_losses.append(loss)
                epoch_losses_fc.append(loss_fc)
                epoch_losses_acc.append(loss_acc)

                if not silent:
                    print(f'Iteration {i+1}/{len(train_loader):<4}loss: {loss.item():0.3f}')

                # Gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = sum(epoch_losses) / len(train_loader)
            epoch_loss_fc = sum(epoch_losses_fc) / len(train_loader)
            epoch_loss_acc = sum(epoch_losses_acc) / len(train_loader)
            tb_writer.add_scalar('stats/train_loss', epoch_loss, epoch)
            tb_writer.add_scalar('stats/train_loss_fc', epoch_loss_fc, epoch)
            tb_writer.add_scalar('stats/train_loss_acc', epoch_loss_acc, epoch)
            print(f"""Epoch {epoch} average loss: {epoch_loss.item():0.3f}, 
                      fc: {epoch_loss_fc.item():0.3f}, 
                      accent: {epoch_loss_acc.item():0.3f}""")

            # validate
            print(f'Testing:')
            model.eval()
            epoch_val_losses = []
            epoch_wer = []
            epoch_acc = []
            with torch.no_grad():
                for data in tqdm(test_loader, total=len(test_loader)): ## ## 
                    inputs, inputs_len, targets, targets_len, target_accents = data
                    total_loss, out_fc, out_acc, output_len = get_loss(*data, mix=mix)
                    val_loss, __, __ = total_loss

                    if DEBUG:
                        print('val loss', val_loss)

                    epoch_val_losses.append(val_loss)

                    wer = check_wer(targets, targets_len, out_fc, output_len)
                    epoch_wer.append(wer)

                    acc = check_acc(target_accents, out_acc)
                    epoch_acc.append(acc)

            epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) ##
            epoch_wer = sum(epoch_wer) / len(epoch_wer)
            epoch_acc = sum(epoch_acc) / len(epoch_acc)

            tb_writer.add_scalar('stats/val_loss', epoch_val_loss, epoch)
            print(f'Average validation loss: {epoch_val_loss.item():0.3f}')

            tb_writer.add_scalar('stats/wer', epoch_wer, epoch)
            print(f'Average wer: {epoch_wer:0.3f}%')

            tb_writer.add_scalar('stats/acc', epoch_acc, epoch)
            print(f'Average acc: {epoch_acc:0.3f}%')

            if epoch_wer < prev_epoch_wer:
                print('New best model found.')
                prev_epoch_wer = epoch_wer     

                torch.save(model.state_dict, f'saved/vac03_sd_{_mix}_{RUN}.pt')
                torch.save(model, f'saved/vac03_fm_{_mix}_{RUN}.pt')

        return model, prev_epoch_val_loss, prev_epoch_wer




    settings = {'mix': [1.],
                'rnn_hidden_size': [800],}
    now_ = time.ctime()
    for _mix in settings['mix']:
        for _rnn_hidden_size in settings['rnn_hidden_size']:
            type_ = 'dev' if DEV else 'train'
            exp_name = f'{type_}_MT_mix-{_mix}_{now_}'

            model = Network(rnn_type=param['rnn_type'], 
                            labels=param['labels'], 
                            rnn_hidden_size=_rnn_hidden_size, 
                            accents=accent_dict,
                            nb_layers=param['num_layers'], #audio_conf=audio_conf,
                            bidirectional=True,
                            DEBUG=DEBUG,)

            if param['cuda']:
                model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'][0])

            print()
            print(f'{"":#<13}')
            print(exp_name)
            print(f'{"":#<13}')

            model, val_loss, wer = train(param, 
                                    model,
                                    train_loader, 
                                    test_loader, optimizer, 
                                    mix=_mix,
                                    exp_name=exp_name)
            
    print('#### END 03 - {RUN} ####')


