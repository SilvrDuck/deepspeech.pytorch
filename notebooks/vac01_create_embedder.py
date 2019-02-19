print('#### START VAC01 ####')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#torch.multiprocessing.set_start_method("spawn")

# Restart from here
DEV = False
EPOCHS = 30

DEBUG = False
NUM_CONCAT = 20

import parameters

# Allows to load modules from parent directory
from time import time
import inspect, sys, os, json
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(inspect.getfile(inspect.currentframe())))))

from pathlib import Path
from os import makedirs

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import create_binarizer, get_accents_counts
from utils import count_parameters
from models.modules import BatchRNN

from tensorboardX import SummaryWriter

import math

from torch.utils.data import DataLoader, Dataset

param = parameters.get_parameters(dev=DEV, epochs=EPOCHS, us_en=False)

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

class KaldiDataset(Dataset):
    """Defines an iterator over the dataset. This class is intended to be used with PyTorch DataLoader"""
    
    def __init__(self, data_path, sample_ids, accent_id_dict):
        
        self.data_path = data_path
        self.accent_id_dict = accent_id_dict
        if isinstance(sample_ids, list):
            self._datafiles = sample_ids
        else:
            with open(sample_ids) as f:
                self._datafiles = [x.strip() for x in f.readlines()]
        
    def __getitem__(self, index):
             
        with open(os.path.join(self.data_path, self._datafiles[index])) as f:
            sample = json.load(f)
        
        target = self.accent_id_dict[extract_num(self._datafiles[index])]
        return torch.FloatTensor(sample), target
                      
    def __len__(self):
        
        return len(self._datafiles)
    
def collate_fn(batch_tot):
    """This function takes list of samples and assembles a batch. It is intended to used in PyTorch DataLoader."""
    res, tar = zip(*batch_tot)

    lens = torch.tensor([len(r) for r in res])
    tar = torch.tensor(tar)
    
    res = nn.utils.rnn.pad_sequence(res, batch_first=True)
    
    __, idx = lens.sort(descending=True)
    
    return res[idx], tar[idx], lens[idx]

class KaldiDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for SpeechDatasets.
        """
        super(KaldiDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        
        
accent_id_dict, accent_dict = make_accent_dict(param['train_manifest'])

train_dataset = KaldiDataset(param['train_kaldi'], 
                              ids_list(param['train_manifest']), 
                              accent_id_dict)

train_loader = KaldiDataLoader(train_dataset, 
                                shuffle=True, 
                                num_workers=0,#param['num_worker'],
                                batch_size=param['batch_size'])

# for data in train_loader:
#     print(data[0])
#     print(data)
#     break

test_dict, __ = make_accent_dict(param['test_manifest'])

test_dataset = KaldiDataset(param['test_kaldi'], 
                              ids_list(param['test_manifest']), 
                              test_dict)

test_loader = KaldiDataLoader(test_dataset, 
                                shuffle=True, 
                                num_workers=param['num_worker'],
                                batch_size=param['batch_size'])


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
        
        self.rnn = rnn_type(input_size, hidden_size, 4, 
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
            
        self._DEBUG = False
        return x, bn
    
model = AccentClassifier(input_size=train_dataset[0][0].size(1), 
                         num_classes=len(accent_dict),
                         rnn_type=param['rnn_type'],
                         hidden_size=param['rnn_hidden_size'],
                         bn_size=param['bn_size'],
                         DEBUG=DEBUG)

if param['cuda']:
    model.cuda()
    

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'][0])

print(model)
print('Model parameters counts:', count_parameters(model))

def train(epochs, 
          model, 
          train_loader, 
          test_loader, 
          optimizer, 
          criterion, 
          silent=True,
          exp_name='__tmp__'):

    # Tensorboard
    tb_path = Path(param['tensorboard_dir']) / exp_name
    makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(tb_path)

    prev_epoch_val_loss = math.inf
    
    ## Train
    for epoch in range(1, param['epochs'] + 1):
        print(f'## EPOCH {epoch} ##')
        print(f'Training:')
        model.train()

        # train
        epoch_losses = []
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, target_accents, lens = data
            inputs = inputs.cuda()
            target_accents = target_accents.cuda()
            lens = lens.cuda()

            # Forward pass
            out, __ = model(inputs, lens)

            loss = criterion(out, target_accents)
            epoch_losses.append(loss)

            if not silent:
                print(f'Iteration {i+1}/{len(train_loader):<4}loss: {loss:0.3f}')

            # Gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(epoch_losses) / len(train_loader)
        tb_writer.add_scalar('stats/train_loss', epoch_loss, epoch)
        print(f'Epoch {epoch} average loss: {epoch_loss:0.3f}')

        # validate
        print(f'Testing:')
        model.eval()
        acc = 0
        tot = 0
        with torch.no_grad():
            epoch_val_losses = []
            for data in tqdm(test_loader, total=len(test_loader)): ## ## 
                inputs, target_accents, lens = data
                inputs = inputs.cuda()
                target_accents = target_accents.cuda()

                out, __ = model(inputs, lens)

                val_loss = criterion(out, target_accents)
                epoch_val_losses.append(val_loss)

                out_arg = np.argmax(out, axis=1).cuda()
                diff = torch.eq(out_arg, target_accents)
                acc += torch.sum(diff)
                tot += len(target_accents)

            acc = acc.item() / tot * 100
            epoch_val_loss = sum(epoch_val_losses) / len(test_loader) ##

        tb_writer.add_scalar('stats/accuracy', acc, epoch)
        print(f'Accent classification accuracy: {acc:0.2f}%')

        tb_writer.add_scalar('stats/val_loss', epoch_val_loss, epoch)
        print(f'Average validation loss: {val_loss:0.3f}')

        if epoch_val_loss < prev_epoch_val_loss:
            print('New best model found.')
            torch.save(model.state_dict, 'saved/03_nbk_sd_vacation.pt')
            torch.save(model, 'saved/03_nbk_fm_vacation.pt')
            
    return model, prev_epoch_val_loss

SILENT = True
best_models = {}

settings = {'rnn_type': [nn.GRU],
            'rnn_hidden_size': [800],
            'bn_size': [256]}

for _rnn_type in settings['rnn_type']:
    for _rnn_hidden_size in settings['rnn_hidden_size']:
        for _bn_size in settings['bn_size']:
            exp_name = f'ACC_CLASS_{_rnn_type}_hidden-{_rnn_hidden_size}_bn-{_bn_size}'

            model = AccentClassifier(input_size=train_dataset[0][0].size(1), 
                                     num_classes=len(accent_dict),
                                     rnn_type=_rnn_type,
                                     hidden_size=_rnn_hidden_size,
                                     bn_size=_bn_size,
                                     DEBUG=DEBUG)

            if param['cuda']:
                model.cuda()

            #from focalloss import FocalLoss
            #criterion = FocalLoss()
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'][0])

            print()
            print(f'{"":#<13}')
            print(exp_name)
            print(f'{"":#<13}')

            model, val_loss = train(param, 
                                    model,
                                    train_loader, 
                                    test_loader, optimizer, 
                                    criterion,
                                    silent=SILENT,
                                    exp_name=exp_name)
print('#### END VAC01 ####')