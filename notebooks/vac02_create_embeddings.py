DEV = True
EPOCHS = 1
DEBUG = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from warpctc_pytorch import CTCLoss
#torch.multiprocessing.set_start_method("spawn")

import parameters

# Allows to load modules from parent directory
from time import time
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

from torch.utils.data import DataLoader, Dataset
from decoder import GreedyDecoder, BeamCTCDecoder



param = parameters.get_parameters(dev=DEV, epochs=EPOCHS, us_en=False)

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
            
class AccentClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 rnn_type,
                 hidden_size,
                 bn_size,
                 DEBUG = False,):
        
        super(AccentClassifier, self).__init__()
        
        self.hidden_size = hidden_size

        self._DEBUG = DEBUG
        
        self.rnn = rnn_type(input_size, hidden_size, 2, 
                            bidirectional=True, 
                            batch_first=True)

#         self.rnn = BatchRNN(input_size, 
#                             hidden_size,
#                             rnn_type=rnn_type,bidirectional=True,
#                             batch_norm=True)
        
        self.bn = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, bn_size),
            nn.ReLU(),
        )
            
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bn_size),
            nn.Linear(bn_size, num_classes),
            nn.ReLU(),
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lens):
        if self._DEBUG:
            print('input x', x.size())

        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        x, __ = self.rnn(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        if self._DEBUG:
            print('after rnn', x.size())
#        
#         x = x.view(x.size(0), x.size(1), 2, self.hidden_size)
        
#         if self._DEBUG:
#             print('after view', x.size())
            
        x = x.mean(dim=1)
        
        if self._DEBUG:
            print('after mean', x.size())
            
        x = self.bn(x)
        bn = x
        
        if self._DEBUG:
            print('after bn', x.size())

        x = self.fc(x)
        
        if self._DEBUG:
            print('after fc', x.size())
            
        x = self.softmax(x)
        
        if self._DEBUG:
            print('after softmax', x.size())
        return x, bn
    
    
accent_id_dict, accent_dict = make_accent_dict(param['train_manifest'])

embedder = AccentClassifier(input_size=40, 
                         num_classes=len(accent_dict),
                         bn_size=100,
                         rnn_type=param['rnn_type'],
                         hidden_size=param['rnn_hidden_size'],
                         DEBUG=DEBUG)
embedder.cuda()

embedder.state_dict = torch.load('saved/03_nbk_sd_vacation.pt')
embedder.eval()

from multiprocessing import Pool
from os import listdir, path



types = ['train', 'test', 'dev']
lens = [195780, 3998, 4079]

for type_, len_ in zip(types, lens):
    source_dir = Path(f'../data/CommonVoice_dataset/kaldi/{type_}-norm')
    target_dir = Path(f'../data/CommonVoice_dataset/large_embeddings/{type_}')

    for file in tqdm(source_dir.iterdir(), total=len_):
        try:
            with open(file, 'r') as f:
                for l in f:
                    a = torch.tensor(eval(l))
                    b = a.view(1, a.size(0), a.size(1))
                    b = b.cuda()
                    __, emb = embedder(b, [b.size(1)])
                target = target_dir / file.stem
                torch.save(emb, target)
        except:
            print('error with ', file)