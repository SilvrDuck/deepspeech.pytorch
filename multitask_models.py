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
    def __init__(self, accents_size, rnn_type=nn.LSTM, labels="abc", 
                rnn_hidden_size=768, nb_layers=5, 
                audio_conf=None, bidirectional=True, context=20,
                side_nb_layers=4, side_rnn_hidden_size=768,
                side_rnn_type=nn.LSTM, nb_shared_layers=2):

        assert nb_shared_layers <= nb_layers,'There must be less shared layers than main layers (nb_shared_layers <= nb_layers).'
        assert nb_shared_layers <= side_nb_layers,'There must be less shared layers than side layers (nb_shared_layers <= side_nb_layers).'

        super(MtAccent, self).__init__(rnn_type=rnn_type, labels=labels, 
                                        rnn_hidden_size=rnn_hidden_size, nb_layers=nb_layers, 
                                        audio_conf=audio_conf, bidirectional=bidirectional, 
                                        context=context)
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
        
        funnel = nn.Linear(side_rnn_hidden_size, accents_size)
        sm = nn.LogSoftmax(dim=0)
        self.logSoftmax = nn.Sequential(funnel, sm)


    @overrides
    def forward(self, x, lenghts):
        lenghts = lenghts.cpu().int()
        output_lenghts = self.get_seq_lens(lenghts)
        x, _ = self.conv(x, output_lenghts)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()

        # Initialize shared layers
        for i in range(self.nb_shared_layers):
            x = self.rnns[i](x, output_lenghts)

        # Rest of side layers
        side_x = self.side_rnns[0](x, output_lenghts)
        for i in range(1, len(self.side_rnns)):
            side_x = self.side_rnns[i](x, output_lenghts)

        side_x = self.logSoftmax(side_x)

        # Rest of main layers
        for i in range(self.nb_shared_layers, len(self.rnns)):
            x = self.rnns[i](x, output_lenghts)

        if not self._bidirectional:
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return x, output_lenghts, side_x
