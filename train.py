import argparse
import json
import os
import time

from utils import Timer, AverageMeter, restricted_float

import numpy as np

import torch
import torch.distributed as dist
from torch.autograd import Variable
from tqdm import tqdm
from torch.nn.modules import CrossEntropyLoss
from warpctc_pytorch import CTCLoss

from data.data_loader import create_binarizer

from data.data_loader import AudioDataLoader, SpectrogramAccentDataset, BucketingSampler, DistributedBucketingSampler
from data.utils import reduce_tensor
from decoder import GreedyDecoder
from models.deepspeech import DeepSpeech
from models.modules import supported_rnns
from models.accent_classification import MtAccent
from multitask_loss import MtLoss


parser = argparse.ArgumentParser(description='DeepSpeech training')

# added arguments
parser.add_argument('--model', default='deepspeech', choices=['deepspeech','mtaccent'], help='Decide which model to use. Available: deepspeech, mtaccent')
parser.add_argument('--side-hidden-layers', default='4', type=int, help='Only for multi-task models. Number of layers in the side network')
parser.add_argument('--side-hidden-size', default='800', type=int, help='Only for multi-task models. Size of the layers in the side network')
parser.add_argument('--side-rnn-type', default='gru', help='Only for multi-task models. Type of the RNN in the side network. rnn|gru|lstm are supported')
parser.add_argument('--bottleneck-size', default='40', type=int, help='Only for multi-task models. Size of the accent features going back in the main net.')
parser.add_argument('--shared-layers', default='2', type=int, help='Only for multi-task models. Number of layers shared by the two networks')
parser.add_argument('--mixing-coef', default='.5', type=restricted_float, help='Only for multi-task models. Coeficient for the losses. Formula is [coef*main_loss + (1-coef)*side_loss.]')
parser.add_argument('--optimizer', default='adam', choices=['adam','sgd'], help='Decide which optimizer to use. Available:adam, sgd')
# base arguments
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.data.cpu().numpy()



if __name__ == '__main__':
    t = Timer()

    args = parser.parse_args()
    args.distributed = args.world_size > 1
    main_proc = True
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder

    loss_results = torch.Tensor(args.epochs)
    main_loss_results = torch.Tensor(args.epochs)
    side_loss_results = torch.Tensor(args.epochs)
    cer_results = torch.Tensor(args.epochs)
    wer_results = torch.Tensor(args.epochs)
    mca_results = torch.Tensor(args.epochs) # mca : missclassified accents

    best_wer = None
    if args.visdom and main_proc:
        from visdom import Visdom

        viz = Visdom()
        opts = dict(title=args.id, ylabel='', xlabel='Epoch', 
            legend=['Loss', 'main_loss', 'side_loss', 'WER', 'CER', 'Accents missclassified'])
        viz_window = None
        epochs = torch.arange(1, args.epochs + 1)
    if args.tensorboard and main_proc:
        os.makedirs(args.log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)

    os.makedirs(save_folder, exist_ok=True)

    accent_binarizer = create_binarizer(args.train_manifest)

    avg_loss, start_epoch, start_iter = 0, 0, 0
    avg_main_loss, avg_side_loss = 0, 0

    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        if args.model == 'deepspeech':
            model = DeepSpeech.load_model_package(package)
        elif args.model == 'mtaccent':
            model = MtAccent.load_model_package(package)
        labels = type(model).get_labels(model)
        audio_conf = type(model).get_audio_conf(model)
        parameters = model.parameters()

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)

        if not args.finetune:  # Don't want to restart training
            if args.cuda:
                model.cuda()
            optimizer.load_state_dict(package['optim_dict'])
            start_epoch = int(package.get('epoch', 1)) - 1  # Index starts at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            avg_main_loss = int(package.get('avg_main_loss', 0))
            avg_side_loss = int(package.get('avg_side_loss', 0))
            loss_results = package['loss_results']
            main_loss_results = package['main_loss_results']
            side_loss_results = package['side_loss_results']
            cer_results = package['cer_results']
            wer_results = package['wer_results']
            mca_results = package['mca_results']
            
            if main_proc and args.visdom and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
                x_axis = epochs[0:start_epoch]
                y_axis = torch.stack(
                    (loss_results[0:start_epoch], 
                    main_loss_results[0:start_epoch], 
                    side_loss_results[0:start_epoch], 
                    wer_results[0:start_epoch], 
                    cer_results[0:start_epoch]), 
                    mca_results[0:start_epoch],
                    dim=1)
                viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=opts,
                )
            if main_proc and args.tensorboard and \
                            package[
                                'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
                for i in range(start_epoch):
                    values = {
                        'Avg Train Loss': loss_results[i],
                        'Avg Main Loss': main_loss_results[i],
                        'Avg Side Loss': side_loss_results[i],
                        'Avg WER': wer_results[i],
                        'Avg CER': cer_results[i],
                        'Accent missclassification': mca_results[i]
                    }
                    tensorboard_writer.add_scalars(args.id, values, i + 1)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        
        if args.side_rnn_type is not None:
            side_rnn_type = args.side_rnn_type.lower()
            assert side_rnn_type in supported_rnns, "side_rnn_type should be either lstm, rnn or gru"

        t.add('creates model')
        if args.model == 'deepspeech':
            model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                                nb_layers=args.hidden_layers,
                                labels=labels,
                                rnn_type=supported_rnns[rnn_type],
                                audio_conf=audio_conf,
                                bidirectional=args.bidirectional)
        elif args.model == 'mtaccent':
            model = MtAccent(accents_size=len(accent_binarizer.classes_),
                                bottleneck_size=args.bottleneck_size,
                                rnn_hidden_size=args.hidden_size,
                                nb_layers=args.hidden_layers,
                                labels=labels,
                                rnn_type=supported_rnns[rnn_type],
                                audio_conf=audio_conf,
                                bidirectional=args.bidirectional,
                                side_nb_layers=args.side_hidden_layers,
                                side_rnn_hidden_size=args.side_hidden_size,
                                side_rnn_type=supported_rnns[side_rnn_type],
                                nb_shared_layers=args.shared_layers)

        parameters = model.parameters()
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    
    if args.model == 'deepspeech':
        criterion = CTCLoss()
    elif args.model == 'mtaccent':
        criterion = MtLoss(CTCLoss(), CrossEntropyLoss(), mixing_coef=args.mixing_coef)

    decoder = GreedyDecoder(labels)


    use_kaldi_features = False

    train_dataset = SpectrogramAccentDataset(audio_conf=audio_conf, 
                                            manifest_filepath=args.train_manifest, 
                                            labels=labels,
                                            normalize=True, 
                                            augment=args.augment, 
                                            accent_binarizer=accent_binarizer,
                                            kaldi=use_kaldi_features)

    test_dataset = SpectrogramAccentDataset(audio_conf=audio_conf, 
                                            manifest_filepath=args.val_manifest, 
                                            labels=labels,
                                            normalize=True, 
                                            augment=False, 
                                            accent_binarizer=accent_binarizer,
                                            kaldi=use_kaldi_features) 


    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)
    
    train_loader = AudioDataLoader(train_dataset,
                                    num_workers=args.num_workers, 
                                    batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)


    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    if args.cuda:
        model.cuda()
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=(int(args.gpu_rank),) if args.rank else None)

    print(model)
    print("Number of parameters: %d" % type(model).get_param_size(model))

    if args.tensorboard and main_proc: # TODO empty scope name problem
        try:
            dummy_inputs = torch.rand(20, 1, 161, 10) # TODO dynamically change size
            if args.cuda:
                 dummy_inputs = dummy_inputs.cuda()
            dummy_size = torch.rand(20)
            tensorboard_writer.add_graph(model, (dummy_inputs, dummy_size), verbose=True)
        except Exception as e:
            print("Exception while creating tensorboard graph:")
            print(e)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    #t.print_report()
    ## TRAIN ##
    t.add('starts epochs')
    display_scaling_coef = None
    for epoch in range(start_epoch, args.epochs):
        t.add(f'begin epoch {epoch}')
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes, target_accents = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            data_time.update(time.time() - end)

            if args.cuda:
                inputs = inputs.cuda()
            t.add(f'epoch {epoch}, batch {i} forward pass')

            if args.model == 'deepspeech':
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                
                loss = criterion(out, targets, output_sizes, target_sizes)
                main_loss, side_loss = torch.tensor(0), torch.tensor(0)
            elif args.model == 'mtaccent':
                if epoch != 0:
                    criterion.toggle_update_coefs(new_value=False)

                out, output_sizes, side_out = model(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                target_accents = np.argmax(target_accents, axis=1) # TODO check if this could be done elsewhere…
                loss = criterion((out, targets, output_sizes, target_sizes), (side_out.cpu(), target_accents))
                main_loss, side_loss = criterion.get_sublosses()

            
            loss = loss / inputs.size(0)  # average the loss by minibatch
            main_loss = main_loss / inputs.size(0)
            side_loss = side_loss / inputs.size(0)

            if args.distributed:
                loss_value = reduce_tensor(loss, args.world_size)[0]
                main_loss_value = reduce_tensor(main_loss, args.world_size)[0]
                side_loss_value = reduce_tensor(side_loss, args.world_size)[0]
            else:
                loss_value = loss.item()
                main_loss_value = main_loss.item()
                side_loss_value = side_loss.item()

            inf = float("inf")
            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            if main_loss_value == inf or main_loss_value == -inf:
                print("WARNING: received an inf main_loss, setting loss value to 0")
                main_loss_value = 0
            if side_loss_value == inf or side_loss_value == -inf:
                print("WARNING: received an inf side_loss, setting side_loss value to 0")
                side_loss_value = 0
            t.add(f'epoch {epoch} backward pass')

            avg_loss += loss_value
            avg_main_loss += main_loss_value
            avg_side_loss += side_loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            # Optimizer step
            optimizer.step()
            t.add('starts computing stuff to print')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                sub_losses = criterion.print_sublosses() if args.model == 'mtaccent' else 'n/a'
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      '(Sub-losses: {sub_losses})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), 
                    batch_time=batch_time, data_time=data_time, 
                    loss=losses, sub_losses=sub_losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(type(model).serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                main_loss_results=main_loss_results,
                                                side_loss_results=side_loss_results,
                                                wer_results=wer_results, 
                                                cer_results=cer_results, 
                                                mca_results=mca_results,
                                                avg_loss=avg_loss,
                                                avg_main_loss=avg_main_loss,
                                                avg_side_loss=avg_side_loss),
                           file_path)
            del loss
            del out
        
        avg_loss /= len(train_sampler)
        avg_main_loss /= len(train_sampler)
        avg_side_loss /= len(train_sampler)

        if display_scaling_coef is None:
            display_scaling_coef = 100. / avg_loss
        display_avg_loss = avg_loss * display_scaling_coef
        display_avg_main_loss = avg_main_loss * display_scaling_coef
        display_avg_side_loss = avg_side_loss * display_scaling_coef
        
        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch

        ## VALIDATION ##
        #t.print_report()
        total_cer, total_wer, total_mca = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, input_percentages, target_sizes, target_accents = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if args.cuda:
                    inputs = inputs.cuda()
                
                if args.model == 'deepspeech':
                    out, output_sizes = model(inputs, input_sizes)
                elif args.model == 'mtaccent':
                    out, output_sizes, side_out = model(inputs, input_sizes)
                    mca = 0

                    for x in range(len(target_accents)):
                        accent_out = np.argmax(torch.exp(side_out[x])) # take exp because we do logsoftmax
                        accent_target = np.argmax(target_accents[x])

                        if accent_out != accent_target:
                            mca += 1
                    total_mca += mca

                decoded_output, _ = decoder.decode(out.data, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer += decoder.cer(transcript, reference) / float(len(reference))
                total_cer += cer
                total_wer += wer
                del out

            wer = total_wer / len(test_loader.dataset)
            cer = total_cer / len(test_loader.dataset)
            
            wer *= 100
            cer *= 100
            
            loss_results[epoch] = display_avg_loss
            main_loss_results[epoch] = display_avg_main_loss
            side_loss_results[epoch] = display_avg_side_loss
            wer_results[epoch] = wer
            cer_results[epoch] = cer
            
            if args.model == 'mtaccent':
                mca = total_mca / len(test_loader.dataset)
                mca *= 100
            else:
                mca = -1 # if the model doesn't use accent, mca doesn't make sense.
            mca_results[epoch] = mca

            mca_print = f'{mca:.3f}' if mca != -1 else 'n/a'
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'
                  'Accent missclassification {mca}\t'.format(epoch + 1, wer=wer, cer=cer, mca=mca_print))

            if args.visdom and main_proc:
                x_axis = epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (loss_results[0:epoch + 1], 
                        main_loss_results[0:epoch + 1],
                        side_loss_results[0:epoch + 1],
                        wer_results[0:epoch + 1], 
                        cer_results[0:epoch + 1], 
                        mca_results[0:epoch + 1]), dim=1)
                if viz_window is None:
                    viz_window = viz.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=opts,
                    )
                else:
                    viz.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=viz_window,
                        update='replace',
                    )
            if args.tensorboard and main_proc:
                values = {
                    'Avg Train Loss': display_avg_loss,
                    'Avg Main Loss': display_avg_main_loss,
                    'Avg Side Loss': display_avg_side_loss,
                    'Avg WER': wer,
                    'Avg CER': cer,
                    'Avg Accent missclassification': mca
                }
                tensorboard_writer.add_scalars(args.id, values, epoch + 1)
                if args.log_params:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                        try: # TODO solve this
                            tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
                        except Exception as e:
                            print('There was an error in tensorboard args.log_params:')
                            print(e)
            if args.checkpoint and main_proc:
                file_path = '%s/deepspeech_%d.pth' % (save_folder, epoch + 1)
                torch.save(type(model).serialize(model, optimizer=optimizer, 
                                                epoch=epoch, 
                                                loss_results=loss_results,
                                                main_loss_results=main_loss_results,
                                                side_loss_results=side_loss_results,
                                                wer_results=wer_results, cer_results=cer_results,
                                                mca_results=mca_results),
                           file_path)
                # anneal lr
                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
                optimizer.load_state_dict(optim_state)
                print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

            if (best_wer is None or best_wer > wer) and main_proc:
                print("Found better validated model, saving to %s" % args.model_path)
                torch.save(type(model).serialize(model, optimizer=optimizer, 
                    epoch=epoch,
                    loss_results=loss_results,
                    main_loss_results=main_loss_results,
                    side_loss_results=side_loss_results,
                    wer_results=wer_results, 
                    cer_results=cer_results), args.model_path)
                best_wer = wer

                avg_loss = 0
            if not args.no_shuffle:
                print("Shuffling batches...")
                train_sampler.shuffle(epoch)
