from os.path import isdir, join, basename
from images_collection import ImagesCollection
from mxnet import ndarray as nd
from os import mkdir
import numpy as np
import mxnet as mx
import random
import math
import time
import cv2

from utils import crop_image, expand_bbox

class ClsIter(mx.io.DataIter):

    def __init__(self, images_collections, cfg, data_name='data', balanced=False,
                 label_name='label', prepared=False, test_cfg=None, output_imnames=False):

        samples = [s for ic in images_collections for s in ic]
        self._samples = samples
        self._cfg = cfg
        self._test_cfg = test_cfg
        self._is_train = test_cfg is None
        self._output_imnames = output_imnames
        self._prepared = prepared
        self._balanced = balanced and self._is_train
        self._max_debug_images = 200
        self._n_debug_images = 0
        self._sel_label = None

        self._batch_size = cfg.TRAIN.BATCH_SIZE if self._is_train else test_cfg.BATCH_SIZE
        self._debug_imgs = cfg.TRAIN.DEBUG_IMAGES if self._is_train else test_cfg.DEBUG_IMAGES
        self._channel_num = 2 if cfg.GRAYSCALE else 4
        self._data_shape = (self._channel_num, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])
        self._label_shape = (1, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])

        self._provide_data = [(data_name, (self._batch_size,) + self._data_shape)]
        self._provide_label = [(label_name, (self._batch_size,))]

        self._init_iter_once()
        self._init_iter()

    def _init_iter_once(self):

        self._indx = 0
        self._samples_num = len(self._samples)

        if self._balanced:
            print('preparing balanced iterator...')
            self._order = list(range(self._samples_num))
        else:
            self._order = list(range(self._samples_num))

    def _init_iter(self):
        self._indx = 0

        if not self._balanced and self._is_train:
            random.shuffle(self._order)

    def __iter__(self):
        return self

    def reset(self):
        self._init_iter()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    @property
    def batch_size(self):
        return self._batch_size

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self._indx >= self._samples_num:
            raise StopIteration()
        else:
            if self._balanced:
                sample = self._samples[self._order[self._indx]]
            else:
                sample = self._samples[self._order[self._indx]]
            self._indx += 1

        return sample

    def next(self):
        """Returns the next batch of data."""
        cfg = self._cfg
        batch_size = self._batch_size
        c, h, w = self._data_shape

        batch_data = nd.zeros((batch_size, h, w, c))
        batch_label = nd.zeros((batch_size,))

        imnames = []
        sample_idx = 0
        try:
            while sample_idx < batch_size:
                sample = self.next_sample()
                im, mask = sample.data
                label_orig = sample.label

                im, mask = self._prepare_data_for_blobs(im, mask)

                # debug images
                imname = basename(sample.id)
                imnames.append(imname)
                if self._debug_imgs:
                    self._debug_image(im, mask, label_orig, imname, self._debug_imgs)

                mask = mask[:,:,np.newaxis]
                input_data = np.concatenate((im, mask), axis=2)
                batch_data[sample_idx] = mx.nd.array(input_data)
                batch_label[sample_idx] = label_orig

                sample_idx += 1
        except StopIteration:
            if not sample_idx:
                raise StopIteration
            batch_data = batch_data[:sample_idx,:,:,:]
            batch_label = batch_label[:sample_idx]

        channel_swap = (0, 3, 1, 2)
        batch_data = nd.transpose(batch_data, axes=channel_swap)

        data_batch = mx.io.DataBatch([batch_data], [batch_label], pad=(batch_size-sample_idx))

        # output imnames for visualizing
        if self._output_imnames:
            return data_batch, imnames
        else:
            return data_batch

    def _prepare_data_for_blobs(self, im, mask):
        """Prepare image and mask for use in a blob."""

        do_aug = self._is_train
        do_crop = not self._prepared
        cfg = self._cfg

        prepare_sz = tuple(cfg.PREPARE_SIZE)
        target_shape = tuple(cfg.INPUT_SHAPE)

        im_w = im.shape[1]
        im_h = im.shape[0]

        # prepare size
        if im.shape[0] != prepare_sz[1] or im.shape[1] != prepare_sz[0]:
            prepare_r = float(prepare_sz[0]) / prepare_sz[1]
            orig_r = float(im.shape[1]) / im.shape[0]

            if orig_r < prepare_r:
                # fit width
                crop_w = im_w
                crop_h = crop_w / prepare_r
            else:
                # fit height
                crop_h = im_h
                crop_w = crop_h * prepare_r

            crop_x = int((im_w - crop_w) / 2.)
            crop_y = int((im_h - crop_h) / 2.)
            crop_w = int(crop_w)
            crop_h = int(crop_h)

            im = im[crop_y:crop_y+crop_h,crop_x:crop_x+crop_w,:]
            mask = mask[crop_y:crop_y+crop_h,crop_x:crop_x+crop_w]

            interpolation = cv2.INTER_AREA if prepare_sz[0] < im.shape[1] else cv2.INTER_LINEAR
            im = cv2.resize(im, prepare_sz, interpolation=interpolation)
            mask = cv2.resize(mask, prepare_sz, interpolation=cv2.INTER_NEAREST)

        # deformation
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.DEFORM_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_DEFORM and to_go:
            # choose axis, 50% x-axis, 50% y-axis
            ratio = np.random.uniform(1., cfg.TRAIN.DISTORT.MAX_DEFORM)
            if np.random.randint(0, 2):
                # x axis
                im = cv2.resize(im, (0, 0), fx=ratio, fy=1., interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (0, 0), fx=ratio, fy=1., interpolation=cv2.INTER_NEAREST)
            else:
                # y axis
                im = cv2.resize(im, (0, 0), fx=1., fy=ratio, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (0, 0), fx=1., fy=ratio, interpolation=cv2.INTER_NEAREST)

        # rotate
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.ROTATE_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_ROTATE and to_go:
            im_w = im.shape[1]
            im_h = im.shape[0]
            angle = np.random.uniform(-cfg.TRAIN.DISTORT.MAX_ROTATE, cfg.TRAIN.DISTORT.MAX_ROTATE)
            M = cv2.getRotationMatrix2D((im_w / 2. + 0.5, im_h / 2. + 0.5), angle, 1)
            im = cv2.warpAffine(im, M, (im_w, im_h))
            mask = cv2.warpAffine(mask, M, (im_w, im_h), cv2.INTER_NEAREST)

        # crop
        im_w = im.shape[1]
        im_h = im.shape[0]

        # scale crop
        crop_shape = (target_shape[0], target_shape[1])
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.SCALE_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_SCALE and to_go:
            min_scale = cfg.TRAIN.DISTORT.SCALE_MIN
            max_scale = cfg.TRAIN.DISTORT.SCALE_MAX
            ratio = np.random.uniform(min_scale, max_scale)
            crop_shape = (int(crop_shape[0] * ratio), int(crop_shape[1] * ratio))

        # check max side
        if crop_shape[0] > im_w or crop_shape[1] > im_h:
            crop_r = float(crop_shape[0]) / crop_shape[1]
            img_r = float(im_w) / im_h
            if crop_r < img_r:
                # fit height
                crop_shape = (int(im_h * crop_r), im_h)
            else:
                # fit width
                crop_shape = (im_w, int(im_w / crop_r))

        cr = [0, 0, crop_shape[0], crop_shape[1]]
        cr[0] = max(int((im_w - crop_shape[0]) * 0.5 + 0.5), 0)
        cr[1] = max(int((im_h - crop_shape[1]) * 0.5 + 0.5), 0)

        # move crop
        if do_aug and cfg.TRAIN.DISTORT.USE_CROP and np.random.randint(0, 2):

            ratio = cfg.TRAIN.DISTORT.CROP_REGION
            ratio = max(0., ratio)
            ratio = min(1., ratio)

            # rectangle where image is cropped
            crop_rect_w = int(crop_shape[0] + ratio * (im_w - crop_shape[0]))
            crop_rect_h = int(crop_shape[1] + ratio * (im_h - crop_shape[1]))

            dif_w = min(im_w, crop_rect_w) - crop_shape[0]
            dif_h = min(im_h, crop_rect_h) - crop_shape[1]

            rand_off_x = np.random.randint(0, dif_w+1)
            rand_off_y = np.random.randint(0, dif_h+1)

            cr[0] = rand_off_x + max(int((im_w - crop_rect_w) * 0.5 + 0.5), 0)
            cr[1] = rand_off_y + max(int((im_h - crop_rect_h) * 0.5 + 0.5), 0)

        if do_crop:
            im = im[cr[1]:cr[1]+cr[3],cr[0]:cr[0]+cr[2],:]
            mask = mask[cr[1]:cr[1]+cr[3],cr[0]:cr[0]+cr[2]]

        im_w = im.shape[1]
        im_h = im.shape[0]

        # mirror
        if do_aug and cfg.TRAIN.DISTORT.USE_FLIP and np.random.randint(0, 2):
            im = np.fliplr(im)
            mask = np.fliplr(mask)

        # add random rectangle
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.OCCL_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_OCCLUSION and to_go:

            im_w = im.shape[1]
            im_h = im.shape[0]

            # select random size
            min_r = cfg.TRAIN.DISTORT.OCCL_MIN_SZ
            max_r = cfg.TRAIN.DISTORT.OCCL_MAX_SZ

            rand_w = np.random.uniform(min_r, max_r)
            rand_h = np.random.uniform(min_r, max_r)

            rect_w = int(rand_w * im_w)
            rect_h = int(rand_h * im_h)

            rand_x = np.random.uniform(-rand_w / 2., 1 - rand_w / 2.)
            rand_y = np.random.uniform(-rand_h / 2., 1 - rand_h / 2.)

            rect_x = int(im_w * rand_x)
            rect_y = int(im_h * rand_y)

            rect_x0 = max([0, rect_x])
            rect_y0 = max([0, rect_y])

            rect_x1 = min([im_w - 1, rect_x + rect_w - 1])
            rect_y1 = min([im_h - 1, rect_y + rect_h - 1])

            rect = [rect_x0, rect_y0, rect_x1 - rect_x0 + 1, rect_y1 - rect_y0 + 1]

            r = np.random.randint(25, 255)
            g = np.random.randint(25, 255)
            b = np.random.randint(25, 255)
            color = (b, g, r)

            im[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2], 0] = b
            im[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2], 1] = g
            im[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2], 2] = r

        # jpeg:
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.JPEG_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_JPEG and to_go:
            # compress
            jpeg_min_q = cfg.TRAIN.DISTORT.JPEG_MIN_QUALITY
            jpeg_max_q = cfg.TRAIN.DISTORT.JPEG_MAX_QUALITY

            jpeg_quality = np.random.uniform(jpeg_min_q, jpeg_max_q)
            im_code = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
            # uncompress
            im = cv2.imdecode(im_code, -1)

            im = im.astype(np.float32)

        # noise
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.NOISE_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_NOISE and to_go:
            sigma = np.random.uniform(0, cfg.TRAIN.DISTORT.NOISE_SIGMA)
            noise_to_add = sigma * 255 * np.random.randn(im.shape[0], im.shape[1], im.shape[2])
            noise_to_add = noise_to_add.astype(np.float32)
            im = im + noise_to_add

        # low resolution
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.LOW_RES_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_LOW_RES and to_go:
            factor = np.random.uniform(cfg.TRAIN.DISTORT.LOW_RES_FACTOR_MIN, cfg.TRAIN.DISTORT.LOW_RES_FACTOR_MAX)
            im_shape = (im.shape[1], im.shape[0]) # (x, y)
            low_shape = (int(im_shape[0] / float(factor)), int(im_shape[1] / float(factor)))
            im = cv2.resize(im, low_shape, interpolation=cv2.INTER_AREA)
            im = cv2.resize(im, im_shape, interpolation=cv2.INTER_LINEAR)

        # brightness:
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.BRIGHTNESS_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_BRIGHTNESS and to_go:

            br_min_a = cfg.TRAIN.DISTORT.BRIGHT_MIN_ALPHA
            br_max_a = cfg.TRAIN.DISTORT.BRIGHT_MAX_ALPHA

            br_min_b = cfg.TRAIN.DISTORT.BRIGHT_MIN_BETA
            br_max_b = cfg.TRAIN.DISTORT.BRIGHT_MAX_BETA

            alpha = np.random.uniform(br_min_a, br_max_a)
            beta = np.random.uniform(br_min_b, br_max_b)
            im = im * alpha + beta

            im = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)[1]
            im = cv2.threshold(im, 0, 0, cv2.THRESH_TOZERO)[1]

        # white balance
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.WHITE_BALANCE_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_WHITE_BALANCE and to_go:

            im_b = im[:, :, 0]
            im_g = im[:, :, 1]
            im_r = im[:, :, 2]

            max_coeff = cfg.TRAIN.DISTORT.WHITE_BALANCE_COEFF
            coeffs = 1 + np.random.uniform(-max_coeff, max_coeff, 3)

            im_b *= coeffs[0]
            im_g *= coeffs[1]
            im_r *= coeffs[2]

            im[:, :, 0] = im_b
            im[:, :, 1] = im_g
            im[:, :, 2] = im_r

            im = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)[1]
            im = cv2.threshold(im, 0, 0, cv2.THRESH_TOZERO)[1]

        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.BLINKS_PROB
        if cfg.TRAIN.DISTORT.USE_BLINKS and to_go:
            im_w = im.shape[1]
            im_h = im.shape[0]

            cx = np.random.uniform(im_w / 4., 3 * im_w / 4.)
            cy = np.random.uniform(im_h / 4., 3 * im_h / 4.)

            xv = np.arange(im_w)
            yv = np.arange(im_h)

            sigma_min = cfg.TRAIN.DISTORT.BLINKS_SIGMA_MIN
            sigma_max = cfg.TRAIN.DISTORT.BLINKS_SIGMA_MAX

            sigma = np.random.uniform(sigma_min, sigma_max)
            xvar = sigma * im_w
            yvar = sigma * im_h

            xvals = 1. / np.sqrt(2 * np.pi) / xvar * np.exp(-(xv - cx) ** 2 / (2 * xvar ** 2))
            yvals = 1. / np.sqrt(2 * np.pi) / yvar * np.exp(-(yv - cy) ** 2 / (2 * yvar ** 2))

            light_min = cfg.TRAIN.DISTORT.BLINKS_LIGHT_MIN
            light_max = cfg.TRAIN.DISTORT.BLINKS_LIGHT_MAX

            coeff = 255 * np.random.uniform(light_min, light_max)

            yvals = yvals[:, np.newaxis]
            H = yvals * xvals
            H = H / H.max()
            H = coeff * H

            # another solution
            im = H + im
            im = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)[1]
            im = cv2.threshold(im, 0, 0, cv2.THRESH_TOZERO)[1]

            # im[:, :, 0] = np.minimum(255, H + im[:, :, 0])
            # im[:, :, 1] = np.minimum(255, H + im[:, :, 1])
            # im[:, :, 2] = np.minimum(255, H + im[:, :, 2])

        # grayscale
        to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.GRAYSCALE_PROB
        if do_aug and cfg.TRAIN.DISTORT.USE_GRAYSCALE and to_go:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # fit to target shape if needed
        if im.shape[0] != target_shape[1] or im.shape[1] != target_shape[0]:
            im = cv2.resize(im, target_shape, interpolation=cv2.INTER_AREA)
        if mask.shape[0] != target_shape[1] or mask.shape[1] != target_shape[0]:
            mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)

        im = im.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)

        im = self._convert_color_space(im, inverse=False)

        return im, mask

    def _convert_color_space(self, im, inverse=False):
        cfg = self._cfg

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

    def _debug_image(self, im, mask, label, imname, output_dir):

        if self._n_debug_images > self._max_debug_images:
            return

        cfg = self._cfg

        if not isdir(output_dir):
            mkdir(output_dir)

        alpha = 0.5
        im_cpy = im.copy()

        # color space
        im_cpy = self._convert_color_space(im_cpy, inverse=True)

        if not cfg.CUT_BY_MASK:
            im_show = np.zeros((im.shape[0], 2 * im.shape[1], 3))
            im_show[:,:im.shape[1],:] = im_cpy
            im_cpy[mask > 0] = 255 * alpha + im_cpy[mask > 0] * (1 - alpha)
            im_show[:,im.shape[1]:,:] = im_cpy
        else:
            im_show = im_cpy
            im_show[mask == 0] = 0
        imname = '{}_{}.jpg'.format(label, np.random.randint(10))
        cv2.imwrite(join(output_dir, imname), im_show)

        self._n_debug_images += 1