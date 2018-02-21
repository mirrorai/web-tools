import sys
from os.path import join, isdir, basename, splitext, dirname, abspath
import numpy as np
import mxnet as mx

def get_model_pretrained(cfg):
    prefix, epoch = cfg.TRAIN.PRETRAINED, cfg.TRAIN.PRETRAINED_EPOCH

    label = mx.sym.var('label')
    sym_model, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    all_layers = sym_model.get_internals()
    net = all_layers['flatten0_output']
    fc = mx.symbol.FullyConnected(data=net, num_hidden=cfg.CLS_NUM, name='fc')

    top = mx.sym.SoftmaxOutput(data=fc, label=label, name='top')
    arg_params = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})

    return top, arg_params, aux_params

def get_model(model_dir, cfg):

    label = mx.sym.var('label')

    current_path = dirname(abspath(__file__))

    sym_model = mx.sym.load(join(model_dir, 'resnet-18-symbol.json'))
    all_layers = sym_model.get_internals()
    net = all_layers['flatten0_output']
    fc = mx.symbol.FullyConnected(data=net, num_hidden=cfg.CLS_NUM, name='fc')
    top = mx.sym.SoftmaxOutput(data=fc, label=label, name='top')

    return top

