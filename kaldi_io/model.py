import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechAutoencoder(nn.Module):
    """Simple seq-to-seq audio sentence autoencoder. Encodes input signal (batch x seqLength x feat_size) into fixed
    vector representation and then tries to decode the same sequence out of it. Must be used with MSE or similar loss."""
    
    def __init__(self, hidden_dim, feat_size, device, dropout=0.3, n_enc_layers=2):
        """
        :param hidden_dim: Hidden size for the RNN
        :param feat_size: Input feature size.
        :param device: torch.device instance.
        :param dropout: applies dropout between stacked RNN layers
        :param n_enc_layers: Number of stacked RNN layers
        """
        
        super(SpeechAutoencoder, self).__init__()
        
        self.feat_size = feat_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = dropout
        self.n_enc_layers = n_enc_layers
    
        self.encoder = nn.GRU(input_size=feat_size,
                              hidden_size=hidden_dim,
                              num_layers=n_enc_layers,
                              dropout=dropout if n_enc_layers > 1 else 0,
                              batch_first=True)

        self.fc = nn.Linear(n_enc_layers * hidden_dim, hidden_dim)
        
        self.decoder = nn.GRUCell(input_size=feat_size, hidden_size=hidden_dim)
  
        self.dec_proj = nn.Linear(hidden_dim, feat_size)

    def init_weights(self, m=0.1):

        self.dec_proj.weight.data.uniform_(-m, m)
        self.dec_proj.bias.data.zero_()

        self.fc.weight.data.uniform_(-m, m)
        self.fc.bias.data.zero_()

    def forward(self, x):

        h_0 = self.encoder(x)[1]

        h_0 = h_0.permute(1, 0, 2).contiguous().view(x.size(0), -1)

        h_0 = F.tanh(self.fc(h_0))

        i_0 = torch.zeros((x.size(0), self.feat_size), device=self.device, dtype=torch.float32)

        outs = []
        for i in range(x.size(1)):

            h_0 = self.decoder(i_0, h_0)
            i_0 = self.dec_proj(h_0)
            outs.append(i_0)

        return torch.stack(outs, 1)
    
    def save(self, path=None):

        if not path:
            path = os.path.join(os.getcwd(), 'model_' + str(time.time()))

        torch.save({
            'hidden_dim': self.hidden_dim,
            'feat_size': self.feat_size,
            'dropout': self.dropout,
            'n_enc_layers': self.n_enc_layers,
            'state_dict': self.state_dict()
        }, path)

    @staticmethod    
    def load(path, load_weights=True):

        model_params = torch.load(path, map_location=lambda storage, loc: storage.cpu())
        
        model = SpeechAutoencoder(model_params['hidden_dim'], 
                                  model_params['feat_size'], 
                                  torch.device('cpu'),
                                  model_params['dropout'],
                                  model_params['n_enc_layers'])
        
        if load_weights:
            model.load_state_dict(model_params['state_dict'])
        
        return model
