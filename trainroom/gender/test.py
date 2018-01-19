import sys
import numpy as np
import mxnet as mx
from os import mkdir, makedirs
from os.path import join, isdir, basename, dirname
from images_collection import ImagesCollection
import argparse
import cv2
from train_config import cfg, cfg_from_file
from iterators import ClsIter

def convert_color_space(im, cfg, inverse=False):
    if not inverse:
        if cfg.GRAYSCALE:
            im = im * cfg.NORM_COEFF
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            # default
            if cfg.SUBTR_MEANS:
                if cfg.MEANS_TYPE == 'pixel':
                    pxmeans = np.array(cfg.PIXEL_MEANS)
                    im -= pxmeans
                elif cfg.MEANS_TYPE == 'image':
                    im -= cfg.MEAN_IMG
                    im /= cfg.STD_IMG
            if cfg.NORM_COEFF != 1.0:
                im = im * cfg.NORM_COEFF

            if cfg.COLOR_SPACE == 'rgb':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        if cfg.GRAYSCALE:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = im / cfg.NORM_COEFF
        else:
            if cfg.COLOR_SPACE == 'rgb':
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            if cfg.NORM_COEFF != 1.0:
                im = im / cfg.NORM_COEFF

            if cfg.SUBTR_MEANS:
                if cfg.MEANS_TYPE == 'pixel':
                    pxmeans = np.array(cfg.PIXEL_MEANS)
                    im += pxmeans
                elif cfg.MEANS_TYPE == 'image':
                    im *= cfg.STD_IMG
                    im += cfg.MEAN_IMG

    return im

def compute_metrics(gt_values, predicted, color_dist, cfg):

    predicted = np.array(predicted)
    gt_values = np.array(gt_values)

    # compute metrics
    mask_rank1 = predicted == gt_values

    mask1 = (predicted - 1) == gt_values
    mask2 = (predicted + 1) == gt_values
    mask3 = (predicted - 2) == gt_values
    mask4 = (predicted + 2) == gt_values

    mask_rank3 = np.logical_or(mask_rank1, mask1)
    mask_rank3 = np.logical_or(mask_rank3, mask2)
    mask_rank5 = np.logical_or(mask_rank3, mask3)
    mask_rank5 = np.logical_or(mask_rank5, mask4)

    correct1_cnt = np.count_nonzero(mask_rank1)
    correct3_cnt = np.count_nonzero(mask_rank3)
    correct5_cnt = np.count_nonzero(mask_rank5)
    total_cnt = predicted.size

    accuracy = float(correct1_cnt) / total_cnt
    accuracy3 = float(correct3_cnt) / total_cnt
    accuracy5 = float(correct5_cnt) / total_cnt

    delta_e = np.mean([color_utils.get_delta_e(color_dist, gt_values[i], predicted[i]) for i in range(predicted.shape[0])])
    cmat = confusion_matrix(gt_values, predicted, range(cfg.CLS_NUM))

    metrics_data = {}
    metrics_data['color error'] = delta_e
    metrics_data['cmat'] = cmat
    metrics_data['accuracy3'] = accuracy3
    metrics_data['accuracy5'] = accuracy5

    return accuracy, metrics_data

def cfg_fix_paths(cfg, exp_dir):
    if cfg.TRAIN.DEBUG_IMAGES:
        cfg.TRAIN.DEBUG_IMAGES = join(exp_dir, cfg.TRAIN.DEBUG_IMAGES)
    if cfg.VALIDATION.DEBUG_IMAGES:
        cfg.VALIDATION.DEBUG_IMAGES = join(exp_dir, cfg.VALIDATION.DEBUG_IMAGES)
    if cfg.TEST.DEBUG_IMAGES:
        cfg.TEST.DEBUG_IMAGES = join(exp_dir, cfg.TEST.DEBUG_IMAGES)

def test(ctx, snapshot, epoch, samples, exp_dir):

    cfg_file = join(exp_dir, 'config.yml')
    cfg_from_file(cfg_file)
    cfg_fix_paths(cfg, exp_dir)
    test_cfg = cfg.TEST

    print('test')
    print(exp_dir)
    print(cfg.TEST.DEBUG_IMAGES)
    print('cfg...')

    if cfg.GPU_ID == -1:
        devices = mx.cpu()
    else:
        devices = mx.gpu(cfg.GPU_ID)

    np.random.seed(cfg.RNG_SEED)
    mx.random.seed(cfg.RNG_SEED)

    label_names = ('label',)
    net = mx.mod.Module.load(snapshot, epoch, label_names=label_names, context=devices)

    collections = [ImagesCollection(samples, cfg)]

    test_iter = ClsIter(collections, cfg, test_cfg=test_cfg)
    net.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
    acc_metric = mx.metric.Accuracy()

    test_iter.reset()
    acc_metric.reset()

    gt_labels_all = None
    pr_labels_all = None
    pr_probs_all = None

    total_batches = int(len(samples) / test_cfg.BATCH_SIZE)
    total_batches += 0 if len(samples) % test_cfg.BATCH_SIZE == 0 else 1

    for nbatch, iter_data in enumerate(test_iter):
        print('testing batch {}..'.format(nbatch))
        progress = float(nbatch) / total_batches
        ctx.update_state(state='PROGRESS', progress=progress, status='Testing batch {}..'.format(nbatch))
        test_batch = iter_data
        net.forward(test_batch, is_train=False)
        pr_probs = net.get_outputs()[0]
        gt_labels = test_batch.label[0]
        acc_metric.update(labels=[gt_labels], preds=[pr_probs])

        gt_labels = gt_labels.asnumpy().astype(np.int32)
        pr_probs = pr_probs.asnumpy() # shape (batch_size, num_classes)
        pr_labels = np.argmax(pr_probs, axis=1) # shape (batch_size, 1)

        gt_labels_all = gt_labels if gt_labels_all is None else np.concatenate((gt_labels_all, gt_labels))
        pr_labels_all = pr_labels if pr_labels_all is None else np.concatenate((pr_labels_all, pr_labels))
        pr_probs_all = pr_probs if pr_probs_all is None else np.concatenate((pr_probs_all, pr_probs))

    res = acc_metric.get()
    name, value = res
    log_str = 'test results'
    print('{}: {}: {}'.format(log_str, name, value))

    return pr_probs_all