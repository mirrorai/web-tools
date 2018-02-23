import sys
from os.path import join, isdir, basename, splitext
import numpy as np
import mxnet as mx
from os import mkdir, getcwd
import argparse
from images_collection import ImagesCollection
from train_config import cfg, cfg_from_file
from iterators import ClsIter
from metrics import ClsMetric
from utils import add_path
from callbacks import StatusUpdater
import time

import logging

def cfg_fix_paths(cfg, exp_dir, model_dir):
    if cfg.TRAIN.DEBUG_IMAGES:
        cfg.TRAIN.DEBUG_IMAGES = join(exp_dir, cfg.TRAIN.DEBUG_IMAGES)
    if cfg.VALIDATION.DEBUG_IMAGES:
        cfg.VALIDATION.DEBUG_IMAGES = join(exp_dir, cfg.VALIDATION.DEBUG_IMAGES)
    if cfg.TEST.DEBUG_IMAGES:
        cfg.TEST.DEBUG_IMAGES = join(exp_dir, cfg.TEST.DEBUG_IMAGES)
    if cfg.TRAIN.PRETRAINED:
        cfg.TRAIN.PRETRAINED = join(model_dir, cfg.TRAIN.PRETRAINED)

def solve(ctx, samples, samples_val, trainroom_dir, exp_dir, resume_model=False, gpu_id=None):

    logs_dir = join(exp_dir, 'logs')
    if not isdir(logs_dir):
        mkdir(logs_dir)

    log_filename = '{}-train.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    log_path = join(logs_dir, log_filename)

    root = logging.getLogger()

    # create file handler which logs messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    # fh.setFormatter(formatter)
    root.addHandler(fh)

    # logging.basicConfig(level=logging.INFO,
    #                     filename=join(logs_dir, log_filename),
    #                     filemode='w')
    logging.info('training log')

    snapshots_dir = join(exp_dir, 'snapshots')
    if not isdir(snapshots_dir):
        mkdir(snapshots_dir)
    snapshots_dir = '{}/'.format(snapshots_dir)

    cfg_file = join(exp_dir, 'config.yml')
    cfg_from_file(cfg_file)
    cfg_fix_paths(cfg, exp_dir, join(trainroom_dir, 'models'))

    if cfg.GPU_ID == -1:
        devices = mx.cpu()
    elif gpu_id is not None:
        devices = mx.gpu(gpu_id)
    else:
        devices = mx.gpu(cfg.GPU_ID)

    np.random.seed(cfg.RNG_SEED)
    mx.random.seed(cfg.RNG_SEED)

    collections = [ImagesCollection(samples, cfg)]
    total_samples = sum([len(c) for c in collections])
    print('total number of samples: {}'.format(total_samples))

    resume_model = False
    if resume_model:
        epoch = int(resume_model)
        sym_model, arg_params, aux_params = mx.model.load_checkpoint(snapshots_dir, epoch)
        resume_model = True
    else:
        with add_path(exp_dir):
            opt_params = __import__('get_optimizer_params')
            model = __import__('get_model')
            if cfg.TRAIN.PRETRAINED:
                sym_model, arg_params, aux_params = model.get_model_pretrained(cfg)
            else:
                sym_model = model.get_model(join(trainroom_dir, 'models'), cfg)
                arg_params = None
                aux_params = None
            initw, optimizer, optimizer_params = opt_params.get_optimizer_params(total_samples, cfg)
            del sys.modules['get_model']
            del sys.modules['get_optimizer_params']

    train_iter = ClsIter(collections, cfg, balanced=cfg.TRAIN.BALANCED)

    batch_size_train = train_iter.batch_size
    if len(samples_val) > 0:
        collections_val = [ImagesCollection(samples_val, cfg)]
        val_iter = ClsIter(collections_val, cfg, test_cfg=cfg.VALIDATION)
    else:
        val_iter = None

    graph_shapes = {}
    graph_shapes['data'] = train_iter.provide_data[0][1]
    graph_shapes['label'] = train_iter.provide_label[0][1]
    graph = mx.viz.plot_network(symbol=sym_model, shape=graph_shapes)
    graph.format = 'png'
    graph.render('{}/graph'.format(exp_dir))

    display = cfg.TRAIN.DISPLAY_ITERS
    snapshot_period = cfg.TRAIN.SNAPSHOT_PERIOD
    epochs_to_train = cfg.TRAIN.EPOCHS

    checkpoint_callback = mx.callback.do_checkpoint(snapshots_dir, snapshot_period)
    epoch_end_callbacks = [checkpoint_callback]

    iters_per_epoch = int(total_samples / batch_size_train)
    iters_per_epoch += 0 if total_samples % batch_size_train == 0 else 1
    batch_end_callback = StatusUpdater(ctx, batch_size_train, epochs_to_train, iters_per_epoch, display)

    eval_metric = mx.metric.Accuracy()
    label_names = ('label',)
    model = mx.mod.Module(symbol=sym_model, label_names=label_names,
        context=devices)
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    if resume_model:
        model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  eval_metric=eval_metric,  # report accuracy during training
                  batch_end_callback = batch_end_callback, # output progress for each 100 data batches
                  epoch_end_callback = epoch_end_callbacks,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  begin_epoch=epoch,
                  num_epoch=epochs_to_train)  # train for at most N dataset passes
    else:
        model.init_params(initw, arg_params=arg_params, aux_params=aux_params, allow_missing=True)
        model.init_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)

        model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  # eval_metric=mx.metric.MSE(),  # report accuracy during training
                  eval_metric=eval_metric,  # report accuracy during training
                  batch_end_callback = batch_end_callback, # output progress for each 100 data batches
                  epoch_end_callback = epoch_end_callbacks,
                  num_epoch=epochs_to_train)  # train for at most 10 dataset passes

    ctx.update_state(state='PROGRESS', progress=1.0, status='Saving model...')

    prefix = snapshots_dir + '/model'
    model.save_checkpoint(prefix, epochs_to_train)

    return prefix, epochs_to_train


