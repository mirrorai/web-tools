# -*- coding:utf-8 -*-
import cv2
import numpy
import base64
import shapely.geometry.polygon


# noinspection PyAbstractClass
class Box(shapely.geometry.polygon.Polygon):
    # Coordinates explanation
    #
    # 0 1 2 3 4 5 <- coordinate lines numbers (this shapely operates)
    # |0|1|2|3|4| <- pixel numbers (that we have on images)
    #
    # l        r
    # V        V
    # +--------+< t
    # |        |
    # |        |
    # +--------+< b

    def __init__(self, mode, l, t, var_x, var_y):
        if mode == 'ltwh':
            r = l + var_x
            b = t + var_y
        elif mode == 'ltrb':
            r = var_x
            b = var_y
        else:
            raise ValueError('Mode `{}` not supported'.format(mode))

        super(Box, self).__init__([(l, t), (l, b), (r, b), (r, t)])

    @property
    def left(self): return self.bounds[0]

    @property
    def right(self): return self.bounds[2]

    @property
    def width(self): return self.right - self.left

    @property
    def top(self): return self.bounds[1]

    @property
    def bottom(self): return self.bounds[3]

    @property
    def height(self): return self.bottom - self.top

    @property
    def ltwh(self): return self.left, self.top, self.width, self.height

    @property
    def ltrb(self): return self.left, self.top, self.right, self.bottom

    def max_side(self): return max(self.width, self.height)

    def json(self, mode='ltwh'):
        if mode == 'ltwh':
            return self.ltwh
        elif mode == 'ltrb':
            return self.ltrb
        elif mode == 'dict':
            return {'x': self.left, 'y': self.top, 'width': self.width, 'height': self.height}
        else:
            raise ValueError('Mode `{}` not supported'.format(mode))

    def iou(self, another):
        return self.intersection(another).area / self.union(another).area

    def __reduce__(self):
        return self.__class__, tuple(['ltrb'] + list(self.ltrb))


def resize_image_wh(img, width, height):
    if width == 0:
        width, height = int(img.shape[1] * float(height) / img.shape[0]), height
    elif height == 0:
        width, height = width, int(img.shape[0] * float(width) / img.shape[1])

    interpolation = cv2.INTER_LANCZOS4

    if 2 * width < img.shape[1] or 2 * height < img.shape[0]:
        interpolation = cv2.INTER_AREA

    return cv2.resize(img, (width, height), interpolation=interpolation)

def resize_image(img, minside, maxside, avoid_upsampling=True):

    img_max_side = max(img.shape[:2])
    img_min_side = min(img.shape[:2])

    if minside == 0:
        # keep original
        img_scale = 1.0
    else:
        img_scale = (float)(minside) / (float)(img_min_side)

    if maxside > 0 and img_scale * img_max_side > maxside:
        img_scale = (float)(maxside) / (float)(img_max_side)

    if avoid_upsampling and img_scale > 1.0:
        return img

    height = (int)(img_scale * img.shape[0])
    width = (int)(img_scale * img.shape[1])

    interpolation = cv2.INTER_LANCZOS4

    if 2 * width < img.shape[1] or 2 * height < img.shape[0]:
        interpolation = cv2.INTER_AREA

    return cv2.resize(img, (width, height), interpolation=interpolation)


def write_image(im, path):
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 100])


def encode_image(im):
    return cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tostring()


def decode_image(im_code):
    im = cv2.imdecode(numpy.fromstring(im_code, dtype=numpy.uint8), -1)
    return im


def decode_image_from_json(im_code):
    return decode_image(base64.b64decode(im_code))


def encode_image_to_json(im):
    return base64.b64encode(encode_image(im))


def read_image(path):
    cv2_img_flag = 1
    img_stream = open(path, 'rb')
    img_array = numpy.asarray(bytearray(img_stream.read()), dtype=numpy.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


def apply_mask_to_image(img, polygon):
    mask = 0.5 * numpy.ones(img.shape, dtype=numpy.float)
    roi_corners = numpy.array([[tuple(map(int, p)) for p in polygon.exterior.coords]], dtype=numpy.int32)
    cv2.fillPoly(mask, roi_corners, (1.0,) * 3)
    return img * mask


def crop(image, box):
    image_width, image_height = image.shape[1], image.shape[0]
    l, t, r, b = map(int, box.bounds)

    # Not using `contains` as its returns true iff object boundaries do not touch at all
    if box.left >= 0 and box.top >= 0 and box.right <= image_width and box.bottom <= image_height:
        # Box is fully inside don't need to create black borders
        return image[t:b, l:r]

    # Copy intersection part and leave other stuff black
    image_box = Box('ltwh', 0, 0, image_width, image_height)
    intersection = image_box.intersection(box)

    cropped_image = numpy.zeros(map(int, (box.height, box.width, image.shape[2])), dtype=image.dtype)
    i_l, i_t, i_r, i_b = map(int, intersection.bounds)
    i_width = i_r - i_l
    i_height = i_b - i_t

    cropped_image[(i_t - t):(i_t - t + i_height), (i_l - l):(i_l - l + i_width), :] = image[i_t:i_b, i_l:i_r, :]
    return cropped_image


def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)


def mean_ssd(image_1, image_2):
    return numpy.mean(numpy.abs(image_1 - image_2))
