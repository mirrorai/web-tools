import cv2
import numpy as np

import sys

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)

def save_image(filename, image, label_im):
    image = image * 255.

    img1 = image[:, :, 0]
    img2 = image[:, :, 1]
    img3 = image[:, :, 2]

    img1[label_im > 0] = 255
    img2[label_im > 0] = 255
    img3[label_im > 0] = 255

    image[:, :, 0] = img1
    image[:, :, 1] = img2
    image[:, :, 2] = img3

    cv2.imwrite(filename, image)

def save_label(filename, label_im):
    label_im[label_im > 0] = 255
    cv2.imwrite(filename, label_im.astype(np.ubyte))

def crop_image(img, bbox):
    x_st = bbox[0]
    y_st = bbox[1]

    x_en = bbox[0] + bbox[2] - 1
    y_en = bbox[1] + bbox[3] - 1

    x_st_pad = int(max(0, -x_st))
    y_st_pad = int(max(0, -y_st))
    x_en_pad = int(max(0, x_en - img.shape[1] + 1))
    y_en_pad = int(max(0, y_en - img.shape[0] + 1))

    x_en = x_en + max(0, -x_st)
    y_en = y_en + max(0, -y_st)
    x_st = max(0, x_st)
    y_st = max(0, y_st)

    if y_st_pad != 0 or y_en_pad != 0 or x_st_pad != 0 or x_en_pad != 0:
        # npad=((y_st_pad, y_en_pad), (x_st_pad, x_en_pad))
        if len(img.shape) == 3:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad, img.shape[2]), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1], :] = img
        elif len(img.shape) == 2:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1]] = img
    else:
        img_pad = img
    img_cropped = img_pad[y_st:y_en+1, x_st:x_en+1]
    return img_cropped

def expand_bbox(bbox, scale):
    bbox_c_x = bbox[0] + (bbox[2] - 1) / 2.0
    bbox_c_y = bbox[1] + (bbox[3] - 1) / 2.0

    bbox_max_side = max(bbox[2], bbox[3])
    bbox_new_size = scale * bbox_max_side

    bbox[0] = int(bbox_c_x - (bbox_new_size - 1) / 2.0)
    bbox[1] = int(bbox_c_y - (bbox_new_size - 1) / 2.0)

    bbox[2] = int(bbox_new_size)
    bbox[3] = int(bbox_new_size)
    return bbox

def add_background(self, im, bg_label, bg):

    MAX_ROTATE = 10
    PREPARE_COEFF = 1.4

    im_w = im.shape[1]
    im_h = im.shape[0]

    l_w = bg_label.shape[1]
    l_h = bg_label.shape[0]

    assert(im_w == l_w and im_h == l_h)

    bg_w = bg.shape[1]
    bg_h = bg.shape[0]

    # prepare size
    # ratio w / h
    bg_r = float(bg_w) / bg_h
    orig_r = float(im_w) / im_h

    if orig_r > bg_r:
        # fit width
        crop_w = bg_w
        crop_h = crop_w / orig_r
    else:
        # fit height
        crop_h = bg_h
        crop_w = crop_h * orig_r

    dif_w = int(bg_w - crop_w)
    dif_h = int(bg_h - crop_h)

    offset_x_center = int(dif_w / 2.)
    offset_y_center = int(dif_h / 2.)

    range_st_w = -int(dif_w / 4.)
    range_en_w = int(dif_w / 4.) + 1
    rand_x = np.random.randint(range_st_w, range_en_w)

    range_st_h = -int(-dif_h / 4.)
    range_en_h = int(dif_h / 4.) + 1
    rand_y = np.random.randint(range_st_h, range_en_h)

    crop_x = offset_x_center + rand_x
    crop_y = offset_y_center + rand_y

    crop_w = int(crop_w)
    crop_h = int(crop_h)

    bg = bg[crop_y:crop_y+crop_h,crop_x:crop_x+crop_w,:]
    prepare_size = (int(PREPARE_COEFF * im_w), int(PREPARE_COEFF * im_h))
    bg = cv2.resize(bg, prepare_size, interpolation=cv2.INTER_LINEAR)

    # random rotation
    angle = np.random.uniform(-MAX_ROTATE, MAX_ROTATE)
    M = cv2.getRotationMatrix2D((bg.shape[1] / 2. + 0.5, bg.shape[0] / 2. + 0.5), angle, 1)
    bg = cv2.warpAffine(bg, M, (bg.shape[1], bg.shape[0]))

    # random crop
    dif_w = bg.shape[1] - im_w
    dif_h = bg.shape[0] - im_h

    offset_x_center = int(dif_w / 2.)
    offset_y_center = int(dif_h / 2.)

    cr = [0, 0, im_w, im_h]

    rand_off_x = offset_x_center + np.random.randint(-int(dif_w / 4.), int(dif_w / 4.))
    rand_off_y = offset_y_center + np.random.randint(-int(dif_h / 4.), int(dif_h / 4.))

    cr[0] = rand_off_x
    cr[1] = rand_off_y

    bg = bg[cr[1]:cr[1]+cr[3],cr[0]:cr[0]+cr[2],:]

    bg_label = bg_label.astype(np.float32)
    bg_label = cv2.GaussianBlur(bg_label,(3,3), 0)

    minv = bg_label.min()
    maxv = bg_label.max()
    if maxv - minv > 0:
        bg_label = (bg_label - minv) / (maxv - minv)

    bg_label = cv2.cvtColor(bg_label, cv2.COLOR_GRAY2BGR)
    img_res = im * (1 - bg_label) + bg * bg_label

    return img_res

