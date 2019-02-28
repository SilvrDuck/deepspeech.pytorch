import torch.nn as nn

parameters = {
    'lr': [3e-4],
    'epochs': 20,
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
    'batch_size': 30,
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
    'lm_path': '../data/language_models/cv-train.lm',
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

def get_parameters(dev=True, epochs=None, us_en=False):
    us_en_ = 'us-en_' if us_en else ''
    
    if dev:
        parameters['train_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}dev.csv'
        parameters['train_kaldi'] = '../data/CommonVoice_dataset/kaldi/dev-norm'
        parameters['train_ivectors'] = '../../AccentedSpeechRecognition/data/dev/ivectors'
        parameters['train_transcripts'] = '../data/CommonVoice_dataset/cv-valid-dev/txt/'
        parameters['train_embeddings_100'] = '../data/CommonVoice_dataset/embeddings_100/dev/cv-valid-dev-sample-'
        parameters['train_embeddings_256'] = '../data/CommonVoice_dataset/embeddings_256/dev/cv-valid-dev-sample-'
        parameters['train_audio'] = '../data/CommonVoice_dataset/cv-valid-dev/wav/'
    else:
        parameters['train_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}train.csv'
        parameters['train_kaldi'] = '../data/CommonVoice_dataset/kaldi/train-norm'
        parameters['train_ivectors'] = '../../AccentedSpeechRecognition/data/train/ivectors'
        parameters['train_transcripts'] = '../data/CommonVoice_dataset/cv-valid-train/txt/'
        parameters['train_embeddings_100'] = '../data/CommonVoice_dataset/embeddings_100/train/cv-valid-train-sample-'
        parameters['train_embeddings_256'] = '../data/CommonVoice_dataset/embeddings_256/train/cv-valid-train-sample-'
        parameters['train_audio'] = '../data/CommonVoice_dataset/cv-valid-train/wav/'
        
    parameters['test_manifest'] = f'../data/CommonVoice_dataset/splits/for_notebooks/{us_en_}test.csv'
    parameters['test_kaldi'] = '../data/CommonVoice_dataset/kaldi/test-norm'
    parameters['test_ivectors'] = '../../AccentedSpeechRecognition/data/test/ivectors'
    parameters['test_transcripts'] = '../data/CommonVoice_dataset/cv-valid-test/txt/'
    parameters['test_embeddings_100'] = '../data/CommonVoice_dataset/embeddings_100/test/cv-valid-test-sample-'
    parameters['test_embeddings_256'] = '../data/CommonVoice_dataset/embeddings_256/test/cv-valid-test-sample-'
    parameters['test_audio'] = '../data/CommonVoice_dataset/cv-valid-dev/wav/'
    
    if epochs is not None:
        parameters['epochs'] = epochs
        
    return parameters