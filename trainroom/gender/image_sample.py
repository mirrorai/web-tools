from abc import abstractmethod
from abc import ABCMeta
import numpy as np
import cv2
from utils import crop_image, expand_bbox

class ImageSample:
    __metaclass__ = ABCMeta
    """Image interface"""

    @property
    @abstractmethod
    def bgr_data(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def label(self):
        pass

    def __hash__(self):
        return hash(self.id)


class SampleWithCache(ImageSample):
    """with cache
    """

    _CACHE_MAX_SIZE = 16 * 1024 * 1024 * 1024 # 16 Gb
    _CACHE = {}
    _CACHE_SIZE = 0

    def __init__(self, image_path, label, cfg):
        self._image_path = image_path
        self._label = label - cfg.LABEL_START
        self._cfg = cfg

    @property
    def data(self):

        if self._image_path in SampleWithCache._CACHE:
            im, mask = SampleWithCache._CACHE[self._image_path]
        else:
            im = cv2.imread(self._image_path)

            if im.shape[:2] != mask.shape[:2]:
                raise Exception('dimensions of image and segment label do not match')

            im, mask = self._prepare_mask(im, mask, self._cfg)

            # prepare size
            prepare_sz = tuple(self._cfg.PREPARE_SIZE)
            if im.shape[0] != prepare_sz[1] or im.shape[1] != prepare_sz[0]:
                prepare_r = float(prepare_sz[0]) / prepare_sz[1]
                orig_r = float(im.shape[1]) / im.shape[0]

                if orig_r < prepare_r:
                    # fit width
                    crop_w = im.shape[1]
                    crop_h = crop_w / prepare_r
                else:
                    # fit height
                    crop_h = im.shape[0]
                    crop_w = crop_h * prepare_r

                crop_x = int((im.shape[1] - crop_w) / 2.)
                crop_y = int((im.shape[0] - crop_h) / 2.)
                crop_w = int(crop_w)
                crop_h = int(crop_h)

                im = im[crop_y:crop_y+crop_h,crop_x:crop_x+crop_w,:]
                mask = mask[crop_y:crop_y+crop_h,crop_x:crop_x+crop_w]

                interp = cv2.INTER_AREA if im.shape[1] > prepare_sz[0] else cv2.INTER_LINEAR
                im = cv2.resize(im, prepare_sz, interpolation=interp)
                mask = cv2.resize(mask, prepare_sz, interpolation=cv2.INTER_NEAREST)

            if SampleWithCache._CACHE_SIZE < SampleWithCache._CACHE_MAX_SIZE:
                SampleWithCache._CACHE[self._image_path] = (im, mask)
                SampleWithCache._CACHE_SIZE += im.shape[0] * im.shape[1] * im.shape[2] * 1 # in bytes (type uint8)
                SampleWithCache._CACHE_SIZE += mask.shape[0] * mask.shape[1] * 1 # in bytes (type uint8)

        return im.copy(), mask.copy()

    @property
    def label(self):
        return self._label

    @property
    def id(self):
        return self._image_path

    @staticmethod
    def reset_cache():
        SampleWithCache._CACHE = {}
        SampleWithCache._CACHE_SIZE = 0

    def _prepare_hair(self, im, mask, cfg):
        mask_index = 1
        work_mask = mask == mask_index
        work_mask = work_mask.astype(np.uint8)
        work_mask = cv2.dilate(work_mask, None, iterations=1)
        return im, work_mask

    def _prepare_skin(self, im, mask, cfg):
        mask_indices = [2, 3, 4, 5, 7, 8]
        work_mask = np.isin(mask, mask_indices)
        return im, work_mask

    def _prepare_race(self, im, mask, cfg):
        mask_indices = [2, 3, 4, 5, 7, 8]
        work_mask = np.isin(mask, mask_indices)
        return im, work_mask

    def _prepare_eyes(self, im, mask, cfg):
        # crop eyes
        mask_index = 3

        eyes_mask = mask == mask_index
        eyes_mask = eyes_mask.astype(np.uint8)

        # connected components
        output = cv2.connectedComponentsWithStats(eyes_mask, 4, cv2.CV_32S)
        # select with max area
        stats = output[2]
        centroids = output[3]

        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, stats.shape[0])]

        if cfg.CROP_METHOD == 0:

            if len(areas) < 2:
                return None, None

            sort_idx = np.argsort(areas)
            sort_idx = sort_idx[::-1]
            max1_idx = sort_idx[0] + 1
            max2_idx = sort_idx[1] + 1

            p1 = centroids[max1_idx]
            p2 = centroids[max2_idx]

            c_x = (p1[0] + p2[0]) / 2.
            c_y = (p1[1] + p2[1]) / 2.

            bbox_w = int(0.66 * im.shape[1])
            bbox_h = int(0.22 * im.shape[0])

            bbox_x = int(c_x - (bbox_w - 1) / 2.)
            bbox_y = int(c_y - (bbox_h - 1) / 2.)

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

        elif cfg.CROP_METHOD == 1:
            # crop one eye
            if len(areas) < 1:
                return None, None

            sort_idx = np.argsort(areas)
            sort_idx = sort_idx[::-1]

            if len(areas) == 1:
                sel_idx = sort_idx[0] + 1
            else:
                sel_rand12 = np.random.randint(0, 2) # 0 or 1
                sel_idx = sort_idx[sel_rand12] + 1

            bbox_x = stats[sel_idx, cv2.CC_STAT_LEFT]
            bbox_y = stats[sel_idx, cv2.CC_STAT_TOP]
            bbox_w = stats[sel_idx, cv2.CC_STAT_WIDTH]
            bbox_h = stats[sel_idx, cv2.CC_STAT_HEIGHT]

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

            factor = float(cfg.PREPARE_SIZE[0]) / cfg.INPUT_SHAPE[0]
            bbox = expand_bbox(bbox, factor)
        else:
            raise NotImplementedError

        # crop image
        eyes_mask = crop_image(eyes_mask, bbox)
        im = crop_image(im, bbox)

        # dilate
        work_mask = cv2.dilate(eyes_mask, None, iterations=1)

        return im, work_mask

    def _prepare_brows(self, im, mask, cfg):
        # crop brows
        mask_index = 2

        brows_mask = mask == mask_index
        brows_mask = brows_mask.astype(np.uint8)

        # connected components
        output = cv2.connectedComponentsWithStats(brows_mask, 4, cv2.CV_32S)
        # select with max area
        stats = output[2]
        centroids = output[3]

        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, stats.shape[0])]

        if cfg.CROP_METHOD == 0:
            if len(areas) < 2:
                # print('no brows on {}'.format(sample.id))
                return None, None

            sort_idx = np.argsort(areas)
            sort_idx = sort_idx[::-1]
            max1_idx = sort_idx[0] + 1
            max2_idx = sort_idx[1] + 1

            p1 = centroids[max1_idx]
            p2 = centroids[max2_idx]

            c_x = (p1[0] + p2[0]) / 2.
            c_y = (p1[1] + p2[1]) / 2.

            bbox_w = int(0.66 * im.shape[1])
            bbox_h = int(0.22 * im.shape[0])

            bbox_x = int(c_x - (bbox_w - 1) / 2.)
            bbox_y = int(c_y - (bbox_h - 1) / 2.)

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
        elif cfg.CROP_METHOD == 1:
            # crop one brow
            if len(areas) < 1:
                # print('no eyes on {}'.format(sample.id))
                return None, None

            sort_idx = np.argsort(areas)
            sort_idx = sort_idx[::-1]

            if len(areas) == 1:
                sel_idx = sort_idx[0] + 1
            else:
                sel_rand12 = np.random.randint(0, 2) # 0 or 1
                sel_idx = sort_idx[sel_rand12] + 1

            bbox_x = stats[sel_idx, cv2.CC_STAT_LEFT]
            bbox_y = stats[sel_idx, cv2.CC_STAT_TOP]
            bbox_w = stats[sel_idx, cv2.CC_STAT_WIDTH]
            bbox_h = stats[sel_idx, cv2.CC_STAT_HEIGHT]

            bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

            factor = float(cfg.PREPARE_SIZE[0]) / cfg.INPUT_SHAPE[0]
            bbox = expand_bbox(bbox, factor)
        else:
            raise NotImplementedError

        # crop image
        brows_mask = crop_image(brows_mask, bbox)
        bgr_data = crop_image(im, bbox)

        # dilate
        num_iter = 1
        work_mask = cv2.dilate(brows_mask, None, iterations=num_iter)

        return bgr_data, work_mask

    def _prepare_lips(self, im, mask, cfg):
        # crop lips
        mask_index = 5

        lips_mask = mask == mask_index
        lips_mask = lips_mask.astype(np.uint8)
        # connected components
        output = cv2.connectedComponentsWithStats(lips_mask, 4, cv2.CV_32S)
        # select with max area
        stats = output[2]
        centroids = output[3]
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, stats.shape[0])]

        if len(areas) == 0:
            # print('no lips on {}'.format(sample.id))
            return None, None
        max_idx = np.argmax(areas) + 1

        p1 = centroids[max_idx]
        c_x = p1[0]
        c_y = p1[1]

        bbox_w = int(0.4 * im.shape[1])
        bbox_h = int(0.2 * im.shape[0])

        bbox_x = int(c_x - (bbox_w - 1) / 2.)
        bbox_y = int(c_y - (bbox_h - 1) / 2.)

        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

        # crop image
        lips_mask = crop_image(lips_mask, bbox)
        bgr_data = crop_image(im, bbox)

        # dilate
        num_iter = 1
        work_mask = cv2.dilate(lips_mask, None, iterations=num_iter)

        return bgr_data, work_mask

    def _prepare_mask(self, im, mask, cfg):

        if cfg.CLS_TYPE == 'skin':
            im, mask = self._prepare_skin(im, mask, cfg)
        elif cfg.CLS_TYPE == 'hair':
            im, mask = self._prepare_hair(im, mask, cfg)
        elif cfg.CLS_TYPE == 'race':
            im, mask = self._prepare_race(im, mask, cfg)
        elif cfg.CLS_TYPE == 'lips':
            im, mask = self._prepare_lips(im, mask, cfg)
        elif cfg.CLS_TYPE == 'eyes':
            im, mask = self._prepare_eyes(im, mask, cfg)
        elif cfg.CLS_TYPE == 'brows':
            im, mask = self._prepare_brows(im, mask, cfg)
        else:
            raise Exception('uknown CLS_TYPE: {}'.format(cfg.CLS_TYPE))

        return im, mask