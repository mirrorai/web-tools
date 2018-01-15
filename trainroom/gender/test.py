import sys
import numpy as np
import mxnet as mx
from os import mkdir, makedirs
from os.path import join, isdir, basename, dirname
from images_collection import ImagesCollection
import argparse
import cv2
from config import cfg, cfg_from_file
from iterators import ClsIter
import color_utils
from sklearn.metrics import confusion_matrix

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

def mark_image(img, gt_label, pr_label, color_map):

    shape_new = (img.shape[0], 2 * img.shape[1], 3)
    marked_img = np.zeros(shape_new, dtype=np.float32)
    if len(img.shape) == 2:
        marked_img[:,0:img.shape[1],:] = img[:,:,np.newaxis]
    else:
        marked_img[:,0:img.shape[1],:] = img

    gt_color = color_utils.hex_to_bgr(color_map[gt_label])
    pr_color = color_utils.hex_to_bgr(color_map[pr_label])

    marked_b = marked_img[:,:,0]
    marked_g = marked_img[:,:,1]
    marked_r = marked_img[:,:,2]

    marked_b[:img.shape[0]/2-1,img.shape[1]+1:] = gt_color[0]
    marked_g[:img.shape[0]/2-1,img.shape[1]+1:] = gt_color[1]
    marked_r[:img.shape[0]/2-1,img.shape[1]+1:] = gt_color[2]

    marked_b[img.shape[0]/2+2:,img.shape[1]+1:] = pr_color[0]
    marked_g[img.shape[0]/2+2:,img.shape[1]+1:] = pr_color[1]
    marked_r[img.shape[0]/2+2:,img.shape[1]+1:] = pr_color[2]

    return marked_img

def visualize_result(images, imnames, gt_labels, pr_labels, color_map, color_dist, output_dir, cfg):

    data = zip(images, imnames, gt_labels, pr_labels)

    for vis_img, imname_orig, gt_label, pr_label in data:

        dst_h = 128
        fy = dst_h / float(vis_img.shape[0])
        fx = fy

        vis_img = cv2.resize(vis_img, (0, 0), fx=fx, fy=fy)

        gt_label = int(gt_label)
        pr_label = int(pr_label)

        marked_img = mark_image(vis_img, gt_label, pr_label, color_map)

        adf = abs(gt_label - pr_label)
        delta_e = color_utils.get_delta_e(color_dist, gt_label, pr_label)
        delta_e = int(delta_e)
        imname = '{}diff_{}gt_{}pr_{}'.format(delta_e, gt_label + cfg.LABEL_START, pr_label + cfg.LABEL_START, imname_orig)
        # imname = '{}pr_{}'.format(pr_label + 1, imname_orig)
        cv2.imwrite(join(output_dir, imname), marked_img)

def get_colors(cfg):
    # init colors
    if cfg.CLS_TYPE == 'skin':
        color_map = color_utils.get_skin_colors()
    elif cfg.CLS_TYPE == 'hair':
        color_map = color_utils.get_hair_colors()
    elif cfg.CLS_TYPE == 'race':
        color_map = color_utils.get_race_colors()
    elif cfg.CLS_TYPE == 'lips':
        color_map = color_utils.get_lips_colors()
    elif cfg.CLS_TYPE == 'eyes':
        color_map = color_utils.get_eyes_colors()
    elif cfg.CLS_TYPE == 'brows':
        color_map = color_utils.get_brows_colors()
    else:
        color_map = None
        print('failed to load colors for {}'.format(cfg.CLS_TYPE))
        exit(-1)

    color_dist = color_utils.color_distances(color_map)
    return color_map, color_dist

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

def test_net(net, epoch, test_cfg, cfg):

    datasets = test_cfg.DATASETS
    batch_size = test_cfg.BATCH_SIZE
    color_map, color_dist = get_colors(cfg)

    debug_dir = None
    if 'DEBUG_IMAGES' in test_cfg:
        debug_dir = test_cfg.DEBUG_IMAGES
        if debug_dir and not isdir(debug_dir):
            makedirs(debug_dir)

    accuracy_values = []

    for indx, dataset in enumerate(datasets):
        image_collection = ImagesCollection(dataset, cfg)

        print('[{}/{}] dataset {}: {} samples'.format(indx + 1,
            len(datasets), dataset.DB_PATH, len(image_collection)))

        output_dir = None
        prepared = False
        err_thres = None
        if 'OUTPUT_DIR' in dataset:
            output_dir = dataset['OUTPUT_DIR']

            if 'APPEND_EPOCH' in dataset:
                output_dir_n = output_dir[:-1] if output_dir.endswith('/') else output_dir
                output_dir_last = basename(output_dir_n)
                output_base_dir = dirname(output_dir_n)
                output_dir = join(output_base_dir, '{}_{}_epoch'.format(output_dir_last, epoch))
                print(output_dir)

            if 'ERROR_THRES' in dataset:
                err_thres = dataset['ERROR_THRES']

            if not isdir(output_dir):
                makedirs(output_dir)


        if 'PREPARED' in dataset:
            prepared = True

        test_iter = ClsIter([image_collection], cfg, test_cfg=test_cfg, output_imnames=True, prepared=prepared)
        net.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)
        acc_metric = mx.metric.Accuracy()

        test_iter.reset()
        acc_metric.reset()

        gt_labels_all = None
        pr_labels_all = None

        for nbatch, iter_data in enumerate(test_iter):
            print('testing batch {}..'.format(nbatch))
            test_batch, imnames = iter_data
            net.forward(test_batch, is_train=False)
            pr_probs = net.get_outputs()[0]
            gt_labels = test_batch.label[0]
            acc_metric.update(labels=[gt_labels], preds=[pr_probs])

            gt_labels = gt_labels.asnumpy().astype(np.int32)
            pr_labels = np.argmax(pr_probs.asnumpy(), axis=1) # shape (batch_size, num_classes)

            gt_labels_all = gt_labels if gt_labels_all is None else np.concatenate((gt_labels_all, gt_labels))
            pr_labels_all = pr_labels if pr_labels_all is None else np.concatenate((pr_labels_all, pr_labels))

            # visualize results
            if output_dir is not None:
                imgs = test_batch.data[0]
                imgs = mx.nd.transpose(imgs, axes=(0, 2, 3, 1))
                imgs = imgs.asnumpy()

                visualize_result(imgs, imnames, gt_labels, pr_labels, color_map, color_dist, output_dir, cfg)

        res = acc_metric.get()
        name, value = res
        log_str = 'test results'
        print('{}: {}: {}'.format(log_str, name, value))

        accuracy, metrics_data = compute_metrics(gt_labels_all, pr_labels_all, color_dist, cfg)

        print('{0}: accuracy rank-1: {1:.5f}'.format(log_str, accuracy))
        print('{0}: accuracy rank-3: {1:.5f}'.format(log_str, metrics_data['accuracy3']))
        print('{0}: accuracy rank-5: {1:.5f}'.format(log_str, metrics_data['accuracy5']))
        print('{0}: color error: {1:.3f}'.format(log_str, metrics_data['color error']))

        if test_cfg.SHOW_CMAT:
            print('confusion matrix:')
            print(metrics_data['cmat'])

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Segmentation network')
    parser.add_argument('--gpus', dest='gpu_ids',
                        help='GPU device id to use 0,1,2', type=str)
    parser.add_argument('--exp_dir', dest='exp_dir', required=True, type=str)
    parser.add_argument('--model_name', dest='model_name', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    model_path = args.model_name
    exp_path = args.exp_dir
    snapshots_dir = join(exp_path, 'snapshots')
    snapshots_dir = '{}/'.format(snapshots_dir)
    # plots_dir = join(exp_path, 'output')

    cfg_file = join(exp_path, 'config.yml')
    cfg_from_file(cfg_file)

    if args.gpu_ids:
        cfg.GPU_IDS = [int(gpu) for gpu in args.gpu_ids.split(',')]

    if len(cfg.GPU_IDS) == 0:
        # use all available GPU
        devices = [mx.gpu(i) for i in range(cfg.GPU_NUM)]
    else:
        devices = [mx.gpu(i) for i in cfg.GPU_IDS]

    np.random.seed(cfg.RNG_SEED)
    mx.random.seed(cfg.RNG_SEED)

    model_name = basename(model_path)
    model_name = model_name[:-7]
    checkpoint_epoch = int(model_name.split('-')[-1])

    label_names = ('label',)
    net = mx.mod.Module.load(snapshots_dir, checkpoint_epoch, label_names=label_names, context=devices)
    test_net(net, checkpoint_epoch, cfg.TEST, cfg)