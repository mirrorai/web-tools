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
    factor_d = 0.5
    optimizer = 'adam'
    # momentum = 0.9
    wd = 0.00001
    epochs_steps = [30, 50, 150]

    iter_steps = [int(s * total_samples / cfg.TRAIN.BATCH_SIZE) for s in epochs_steps]
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
