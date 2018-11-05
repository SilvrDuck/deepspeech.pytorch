import time
import time

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from data_loader import *
from model import SpeechAutoencoder

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device: ', device)


#partition = split_data('data/wsj', [0.9, 0.1, 0])

#ds_trn = SpeechDataset('data/wsj', partition['train'])
#ds_dev = SpeechDataset('data/wsj', partition['dev'])

ds_trn = SpeechDataset('data/wsj', 'data/wsj_splits/trn_ids')
ds_dev = SpeechDataset('data/wsj', 'data/wsj_splits/dev_ids')

trn_batch_size = 32
dev_batch_size = 128
hidden_dim = 128
n_epochs = 1024
print_interval = 10
eval_interval = 50


data_loader = DataLoader(ds_trn, batch_size=trn_batch_size, sampler=RandomSampler(ds_trn), collate_fn=collate_fn, num_workers=4, pin_memory=True)
dev_data_loader = DataLoader(ds_dev, batch_size=dev_batch_size, sampler=SequentialSampler(ds_dev), collate_fn=collate_fn, num_workers=4, pin_memory=True)

#model = SpeechAutoencoder.load('models/model_epoch_30').to(device)
#model.device = device
model = SpeechAutoencoder(hidden_dim, 40, device).to(device)

optimizer = optim.Adam(model.parameters())

loss_criterion = nn.MSELoss(size_average=True)




def evaluate(model, data_loader):

    with torch.no_grad():

        model.eval()

        loss = 0
        for i, x in enumerate(data_loader):
            inp = x.cuda(device=device, non_blocking=True)
            y = model(inp)
            loss += F.mse_loss(inp, y).item()

        model.train()

        return loss / i


model.train()
start_time = time.time()
for i_epoch in range(n_epochs):

    total_loss = 0
    for i_batch, sample_batched in enumerate(data_loader):
        

        inp = sample_batched

        
        inp = inp.cuda(device=device, non_blocking=True)


        optimizer.zero_grad()

        pred = model(inp)

        loss = loss_criterion(pred, inp)
    
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    
        optimizer.step()

        total_loss += loss.item()
    
        if (i_batch + 1) % print_interval == 0:

            elapsed = time.time() - start_time
            print('Epoch: {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(
                i_epoch + 1, i_batch + 1, len(ds_trn) // trn_batch_size, elapsed * 1000 / print_interval, total_loss / print_interval))

            start_time = time.time()

            total_loss = 0

        if (i_batch + 1) % eval_interval == 0:
            eval_loss = evaluate(model, dev_data_loader)
            print("===== Dev Loss: %.5f =====" % eval_loss)

    model.save('models/model_epoch_%d' % (i_epoch + 1))
