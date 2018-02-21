import sys
from os.path import join, isdir, basename, splitext
import numpy as np
import mxnet as mx

def get_optimizer_params(total_samples, cfg):

    initw = mx.initializer.Mixed(['bias', '.*'],
        [mx.init.Zero(), mx.initializer.Xavier(factor_type='in', magnitude=2.34)])

    factor_d = None
    optimizer = None
    wd = None
    momentum = None

    base_lr = 1e-4
    factor_d = 0.1
    optimizer = 'adam'
    # momentum = 0.9
    wd = 0.00001
    epochs_steps = [30, 45]

    iters_per_epoch = int(total_samples / cfg.TRAIN.BATCH_SIZE)
    iters_per_epoch += 0 if total_samples % cfg.TRAIN.BATCH_SIZE == 0 else 1

    iter_steps = [int(st * iters_per_epoch) for st in epochs_steps]

    lr_sch = mx.lr_scheduler.MultiFactorScheduler(iter_steps, factor=factor_d)

    optimizer_params = []
    if base_lr:
        optimizer_params.append(('learning_rate', base_lr))
    if lr_sch:
        optimizer_params.append(('lr_scheduler', lr_sch))
    if momentum:
        optimizer_params.append(('momentum', momentum))
    if wd:
        optimizer_params.append(('wd', wd))

    optimizer_params = tuple(optimizer_params)

    return initw, optimizer, optimizer_params
