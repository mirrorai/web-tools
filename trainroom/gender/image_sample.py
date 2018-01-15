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
        self._label = label
        self._cfg = cfg

    @property
    def data(self):

        if self._image_path in SampleWithCache._CACHE:
            im = SampleWithCache._CACHE[self._image_path]
        else:
            im = cv2.imread(self._image_path)

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

                interp = cv2.INTER_AREA if im.shape[1] > prepare_sz[0] else cv2.INTER_LINEAR
                im = cv2.resize(im, prepare_sz, interpolation=interp)

            if SampleWithCache._CACHE_SIZE < SampleWithCache._CACHE_MAX_SIZE:
                SampleWithCache._CACHE[self._image_path] = im
                SampleWithCache._CACHE_SIZE += im.shape[0] * im.shape[1] * im.shape[2] * 1 # in bytes (type uint8)

        return im.copy()

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