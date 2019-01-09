from timeit import default_timer as timer
import argparse
import numpy as np


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f'{x} not in range [0.0, 1.0]')     
    return x

class Timer():

    def __init__(self):
        self.events = []
        self.start_time = timer()
        self.append_event('Start', 0., 0.)

    def reset_time(self, event='Timer reset'):
        self.append_event(event, 0., 0.)
        self.start_time = timer()

    def get_time(self):
        return timer()

    def get_previous_time(self):
        return self.events[-1]['Absolute']

    def append_event(self, event, relative, absolute):
        self.events.append(
            {'Event': event, 'Relative': relative, 'Absolute': absolute}
        )

    def add(self, event):
        now = timer()
        absolute = now - self.start_time
        relative = absolute - self.get_previous_time()
        self.append_event(event, relative, absolute)
        

    def print_report(self, decimals=3):
        cropped = lambda x: f'{x:.{decimals}f}' if isinstance(x, float) else str(x)

        col_width = {}
        keys = self.events[0].keys()
        for k in keys:
            items_per_col = [len(cropped(row[k])) for row in self.events]
            items_per_col.append(len(k))
            col_width[k] = max(items_per_col) + 2
            
        print('### Timer report ###')
        print(''.join(k.rjust(col_width[k]) for k in keys))

        for row in self.events:
            print(''.join(cropped(v).rjust(col_width[k]) for k, v in row.items()))
        print('### End report   ###')
        

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params