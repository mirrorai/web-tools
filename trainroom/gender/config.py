"""Config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.DATASETS = []

# Minibatch size
__C.TRAIN.BATCH_SIZE = 128

# Number of total train epochs
__C.TRAIN.EPOCHS = 100
__C.TRAIN.PRETRAINED = ''
__C.TRAIN.PRETRAINED_EPOCH = 0
__C.TRAIN.BALANCED = False

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_PERIOD = 10
__C.TRAIN.DISPLAY_ITERS = 20

__C.TRAIN.DISTORT = edict()

__C.TRAIN.DISTORT.USE_ROTATE = 1
__C.TRAIN.DISTORT.ROTATE_PROB = 0.5
__C.TRAIN.DISTORT.MAX_ROTATE = 10

__C.TRAIN.DISTORT.USE_DEFORM = 1
__C.TRAIN.DISTORT.DEFORM_PROB = 0.5
__C.TRAIN.DISTORT.MAX_DEFORM = 1.1

__C.TRAIN.DISTORT.USE_FLIP = 1

__C.TRAIN.DISTORT.USE_CROP = 1
__C.TRAIN.DISTORT.CROP_REGION = 1.0

__C.TRAIN.DISTORT.USE_SCALE = 1
__C.TRAIN.DISTORT.SCALE_PROB = 0.5
__C.TRAIN.DISTORT.MAX_SCALE = 0.05
__C.TRAIN.DISTORT.SCALE_MIN = 0.95
__C.TRAIN.DISTORT.SCALE_MAX = 1.05

__C.TRAIN.DISTORT.USE_GRAYSCALE = 0
__C.TRAIN.DISTORT.GRAYSCALE_PROB = 0.1

__C.TRAIN.DISTORT.USE_BRIGHTNESS = 1
__C.TRAIN.DISTORT.BRIGHTNESS_PROB = 0.25
__C.TRAIN.DISTORT.BRIGHT_MIN_ALPHA = 0.95
__C.TRAIN.DISTORT.BRIGHT_MAX_ALPHA = 1.05

__C.TRAIN.DISTORT.BRIGHT_MIN_BETA = 0.
__C.TRAIN.DISTORT.BRIGHT_MAX_BETA = 0.

__C.TRAIN.DISTORT.USE_WHITE_BALANCE = 0
__C.TRAIN.DISTORT.WHITE_BALANCE_PROB = 0.5
__C.TRAIN.DISTORT.WHITE_BALANCE_COEFF = 0.01

__C.TRAIN.DISTORT.USE_BLINKS = 0
__C.TRAIN.DISTORT.BLINKS_PROB = 0.25
__C.TRAIN.DISTORT.BLINKS_SIGMA_MIN = 0.27
__C.TRAIN.DISTORT.BLINKS_SIGMA_MAX = 0.61
__C.TRAIN.DISTORT.BLINKS_LIGHT_MIN = 0.3
__C.TRAIN.DISTORT.BLINKS_LIGHT_MAX = 0.4

__C.TRAIN.DISTORT.USE_JPEG = 1
__C.TRAIN.DISTORT.JPEG_PROB = 0.25
__C.TRAIN.DISTORT.JPEG_MIN_QUALITY = 12
__C.TRAIN.DISTORT.JPEG_MAX_QUALITY = 20

__C.TRAIN.DISTORT.USE_NOISE = 0
__C.TRAIN.DISTORT.NOISE_PROB = 0.5
__C.TRAIN.DISTORT.NOISE_SIGMA = 0.1

__C.TRAIN.DISTORT.USE_LOW_RES = 0
__C.TRAIN.DISTORT.LOW_RES_PROB = 0.25
__C.TRAIN.DISTORT.LOW_RES_FACTOR_MIN = 1.5
__C.TRAIN.DISTORT.LOW_RES_FACTOR_MAX = 2.5

__C.TRAIN.DISTORT.USE_OCCLUSION = 0
__C.TRAIN.DISTORT.OCCL_PROB = 0.0
__C.TRAIN.DISTORT.OCCL_MIN_SZ = 0.2
__C.TRAIN.DISTORT.OCCL_MAX_SZ = 0.6

__C.TRAIN.DISTORT.BG_AUGMENT_PROB = 0.0

__C.TRAIN.DISTORT.USE_BG_NEGATIVE = 0
__C.TRAIN.DISTORT.BG_NEGATIVE_PROB = 0.0
__C.TRAIN.BG_AUGMENT_DATASETS = []

__C.TRAIN.DEBUG_IMAGES = ''

#
# Testing options
#
__C.TEST = edict()

__C.TEST.DATASETS = []

# Minibatch size
__C.TEST.BATCH_SIZE = 128
__C.TEST.DEBUG_IMAGES = ''
__C.TEST.SHOW_CMAT = True
__C.TEST.USE_AUGMENTATION = False
__C.TEST.AUGMENTATION_COUNT = 5

#
# Validation options
#
__C.VALIDATION = edict()

__C.VALIDATION.DATASETS = []
__C.VALIDATION.INTERVAL = 1000
__C.VALIDATION.DEBUG_IMAGES = ''

# Minibatch size
__C.VALIDATION.BATCH_SIZE = 128
__C.VALIDATION.SHOW_CMAT = False

__C.VALIDATION.USE_AUGMENTATION = False
__C.VALIDATION.AUGMENTATION_COUNT = 0

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# __C.PIXEL_MEANS = np.array([[[104.00699, 116.66877, 122.67892]]])
__C.PIXEL_MEANS = [104.00699, 116.66877, 122.67892]
__C.MEAN_IMG = np.array([])
__C.MEAN_STD = np.array([])
__C.IMAGE_MEANS_PATH = ''
__C.IMAGE_STD_PATH = ''
__C.MEANS_TYPE = 'pixel' # or 'image'
__C.SUBTR_MEANS = False
__C.GRAYSCALE = False
__C.COLOR_SPACE = 'rgb' # can be bgr

__C.TYPE = 'all'
__C.CLS_TYPE = 'skin' # can be hair, race
__C.CLS_NUM = 9
__C.CROP_METHOD = 0
__C.METHOD_VERSION = 0
__C.CUT_BY_MASK = False

__C.LABEL_START = 1

# For reproducibility
__C.RNG_SEED = 3

# Default GPU device id
__C.GPU_ID = 0
__C.GPU_NUM = 4

# Prepare size
__C.PREPARE_SIZE = [200, 200]
__C.INPUT_SHAPE = [160, 160]
__C.NORM_COEFF = 1.0 / 256.0

def get_output_dir(suffix, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'exps', __C.EXP_DIR, 'output', suffix))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value