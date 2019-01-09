import torch.nn as nn

parameters = {
    'lr': [3e-4],
    'epochs': 3,
    'sample_rate': 16_000,
    'window_size': .02,
    'window_stride': .01,
    'window': 'hamming',
    'noise_dir': None,
    'noise_prob': 0.4,
    'noise_min': 0.0,
    'noise_max': 0.5,
    'labels': "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    'max_norm': 400,
    'save_folder': '../models/best',
    'augment': True,
    'batch_size': 40,
    'num_worker': 4,
    'cuda': True,
    'tensorboard_dir': '../runs/',
    'rnn_type': nn.GRU,
    'rnn_hidden_size': 40,
    'bn_size': 100,
}

parameters['audio_conf'] = {'sample_rate': parameters['sample_rate'],
                           'window_size': parameters['window_size'],
                           'window_stride': parameters['window_stride'],
                           'window': parameters['window'],
                           'noise_dir': parameters['noise_dir'],
                           'noise_prob': parameters['noise_prob'],
                           'noise_levels': (parameters['noise_min'], 
                                            parameters['noise_max'])}

def get_parameters(dev=True, epochs=None, us_en=False):
    us_en_ = 'us-en_' if us_en else ''
    
    if dev:
        parameters['train_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}dev.csv'
        parameters['train_kaldi'] = '../data/CommonVoice_dataset/kaldi/dev-norm'
    else:
        parameters['train_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}train.csv'
        parameters['train_kaldi'] = '../data/CommonVoice_dataset/kaldi/train-norm'
        
    parameters['test_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}test.csv'
    parameters['test_kaldi'] = '../data/CommonVoice_dataset/kaldi/test-norm'
    
    if epochs is not None:
        parameters['epochs'] = epochs
        
    return parameters