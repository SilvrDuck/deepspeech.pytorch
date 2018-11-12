import math
import torch
import torch.nn as nn

from collections import OrderedDict

from modules import MaskConv, SequenceWise, BatchRNN, InferenceBatchSoftmax, Lookahead, \
                    supported_rnns, supported_rnns_inv

class LeveragingNative(nn.Module):
    def __init__(self,
                rnn_type=nn.LSTM,
                labels='abc',
                feedforward_size=500,
                nb_shared_ff=2, nb_l1_ff=2, nb_l2_ff=2,
                rnn_size=300,
                nb_shared_rnn=1, nb_l1_rnn=1, nb_l2_rnn=1,
                audio_conf=None):

        super(LeveragingNative, self).__init__()

        shared_ffs = []
        for x in range(nb_shared_ff):
            ff = nn.Sequential(
                nn.BatchNorm1d(feedforward_size),
                nn.Linear(rnn_size, rnn_size))
            shared_ffs.append(('%d' % (x + 1), ff))
        self.shared_ffs = nn.Sequential(OrderedDict(shared_ffs))

        shared_rnns = []
        for x in range(nb_shared_rnn):
            rnn = BatchRNN(input_size=rnn_size, hidden_size=rnn_size, 
                            rnn_type=rnn_type, bidirectional=True)
            shared_rnns.append(('%d' % (x + 1), rnn))
        self.shared_rnns = nn.Sequential(OrderedDict(shared_rnns))







        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()