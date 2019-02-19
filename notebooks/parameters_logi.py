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
    'batch_size': 20,
    'num_worker': 4,
    'cuda': True,
    'tensorboard_dir': '../runs/',
    'rnn_type': nn.GRU,
    'rnn_hidden_size': 800,
    'bn_size': 512,
    'num_layers': 5,
    'num_layers_main': 4,
    'num_layers_head': 2,
    'num_layers_acc': 4,
    'lm_path': '../data/language_models/lnn.lm',
    'mix': .9,
    'augment': True,
    'normalize': True,
    'embedding_size': 256,
}

parameters['audio_conf'] = {'sample_rate': parameters['sample_rate'],
                           'window_size': parameters['window_size'],
                           'window_stride': parameters['window_stride'],
                           'window': parameters['window'],
                           'noise_dir': parameters['noise_dir'],
                           'noise_prob': parameters['noise_prob'],
                           'noise_levels': (parameters['noise_min'], 
                                            parameters['noise_max'])}

def get_parameters(native=False, dev=True, epochs=None, us_en=False):
    print('Warning: No train set available for LogiDataset. Replaced by the test set.')
    if us_en:
        print('Warning: No us_en parameter for LogiDataset.')
        
    if native:
        parameters['test_manifest'] = f'../data/LogiDataset/splits/for_notebooks/native.manifest'
    else:
        parameters['test_manifest'] = f'../data/LogiDataset/splits/for_notebooks/lnn.manifest'
        
    parameters['test_kaldi'] = '../data/LogiDataset/kaldi/norm'
    parameters['test_ivectors'] = '../data/LogiDataset/kaldi/ivectors/'
    parameters['test_transcripts'] = '../data/LogiDataset/txt/'
    parameters['test_embeddings_100'] = '../data/LogiDataset/embeddings_100/'
    parameters['test_embeddings_256'] = '../data/LogiDataset/embeddings_256/'
    parameters['test_audio'] = '../data/LogiDataset/wav/'
        
    parameters['train_manifest'] = parameters['test_manifest']
    parameters['train_kaldi'] = parameters['test_kaldi']
    parameters['train_ivectors'] = parameters['test_ivectors']
    parameters['train_transcripts'] = parameters['test_transcripts']
    parameters['train_embeddings_100'] = parameters['test_embeddings_100']
    parameters['train_embeddings_256'] = parameters['test_embeddings_256']
    parameters['train_audio'] = parameters['test_audio']
    
    if epochs is not None:
        parameters['epochs'] = epochs
        
    return parameters