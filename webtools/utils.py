# -*- coding:utf-8 -*-

from __future__ import unicode_literals

import locale
locale.setlocale(locale.LC_ALL, '')

import sys
import arrow
import re
from datetime import timedelta
import errno
import os
import os.path
import tarfile
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import hashlib

from flask import abort, redirect, request

from . import app, opencv
import reannotation.models
from sqlalchemy import or_, and_
from operator import itemgetter, attrgetter

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)

class IllegalApiUsage(Exception):
    pass

def parse_timing_parameters():
    def parse_timestamp(parameter_name):
        utc_timestamp = request.args.get(parameter_name, None, type=int)
        return None if utc_timestamp is None else arrow.Arrow.utcfromtimestamp(utc_timestamp)
    start_utc_datetime = parse_timestamp('start_utc_timestamp')
    end_utc_datetime = parse_timestamp('end_utc_timestamp')

    frequency = request.args.get('frequency', None, type=int)
    if frequency is not None:
        frequency = timedelta(seconds=frequency)

    points = request.args.get('points', None, type=int)

    return {
        'start_utc_datetime': start_utc_datetime,
        'end_utc_datetime': end_utc_datetime,
        'frequency': frequency,
        'points': points
    }

def apply_timing_filters(query, timestamp, params):
    if params['start_utc_datetime'] is not None:
        query = query.filter(timestamp >= params['start_utc_datetime'])
    if params['end_utc_datetime'] is not None:
        query = query.filter(timestamp <= params['end_utc_datetime'])
    return query

def parse_min_max_detections_parameters():
    return {
        'min_detections': request.args.get('min_detections', None, type=int),
        'max_detections': request.args.get('max_detections', None, type=int)
    }

def apply_min_max_detections_filters(query, detections_count, params):
    if params['min_detections'] is not None:
        query = query.filter(detections_count >= params['min_detections'])
    if params['max_detections'] is not None:
        query = query.filter(detections_count <= params['max_detections'])
    return query

def read_page_query(default_per_page=20, max_per_page=100):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', default_per_page, type=int)
    per_page = max(1, min(max_per_page, per_page))

    return page, per_page

def preprocess_paged_query(query, endpoint):
    # BUG: redirect may loose other parameters
    page, per_page = read_page_query()

    if page < 1:
        return True, redirect(endpoint + '?per_page={}&page={}'.format(per_page, 1))

    query = query.paginate(page, per_page, False)
    if query.pages == 0:
        return False, query

    if page > query.pages:
        return True, redirect(endpoint + '?per_page={}&page={}'.format(per_page, query.pages))

    return False, query

def get_image(data_json, key):
    im_code = data_json.get(key, None)
    if im_code is None:
        app.logger.warning('Code of the image to decode is None')
        abort(422)
    try:
        image = opencv.decode_image_from_json(im_code)
        return image
    except:
        app.logger.warning('Exception on decoding image')
        abort(422)

def median(lst):
    lst = sorted(lst)
    if len(lst) < 1:
            return None
    return lst[len(lst) / 2]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# http://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        # noinspection PyShadowingBuiltins
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(root[len(path):], file))

# http://code.activestate.com/recipes/576714-extract-a-compressed-file/
def extract_archive(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError('Could not extract `{}` as no appropriate extractor is found'.format(path))

    f = opener(path, mode)
    try:
        f.extractall(to_directory)
    finally:
        f.close()

def camelcase_to_snakecase(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snakecase_to_camelcase(value):
    def camelcase():
        yield str.lower
        while True:
            yield str.capitalize

    c = camelcase()
    return ''.join(c.next()(x) if x else '_' for x in value.split("_")).capitalize()

def list_images(base_dir):
    images_list = []
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    for f in os.listdir(base_dir):
        if not os.path.isfile(os.path.join(base_dir, f)):
            continue
        filext = os.path.splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list

def list_jsons(base_dir):
    out_list = []
    valid_exts = ['.json']
    for f in os.listdir(base_dir):
        if not os.path.isfile(os.path.join(base_dir, f)):
            continue
        filext = os.path.splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        out_list.append(f)
    return out_list

def list_subdirs(base_dir, pattern=None):
    subdirs = list()
    for child in os.listdir(base_dir):
        if pattern is not None and not re.match(pattern, child):
            continue
        subdir_path = os.path.join(base_dir, child)
        if os.path.isdir(subdir_path):
            subdirs.append(child)
    return subdirs

def list_marking_files(base_dir):
    marking_list = []
    valid_exts = ['.xml', '.json']
    for f in os.listdir(base_dir):
        if not os.path.isfile(os.path.join(base_dir, f)):
            continue
        filext = os.path.splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        marking_list.append(f)
    return marking_list

def validate_size(img):
    MAX_SIDE_SCALE = 2400
    MIN_SIDE_SCALE = 1800

    img_sz = opencv.resize_image(img, MIN_SIDE_SCALE, MAX_SIDE_SCALE)
    im_scale = float(img_sz.shape[0]) / float(img.shape[0])

    return img_sz, im_scale

def check_image(img_path):
    MIN_IMG_SIZE = 64
    if not os.path.exists(img_path):
        return -1

    try:
        img = opencv.read_image(img_path)
    except:
        return -2

    if img is not None:
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_min_size = min(img_w, img_h)
        # try to read pixel
        try:
            pix = img[0]
        except:
            return -2
    else:
        return -2

    if img_min_size < MIN_IMG_SIZE:
        return -3

    return 0

def find_fp_fn(score_map_file=None, level_fp=None, level_fn=None, only_classes=None):

    if level_fp is None:
        conf_level_fp = 0.7
    else:
        conf_level_fp = level_fp

    if level_fn is None:
        conf_level_fn = 0.4
    else:
        conf_level_fn = level_fn

    brands = reannotation.models.Brand.query.all()
    if only_classes is not None:
        brands = [b for b in brands if b.brand_class in only_classes]

    reannotation.models.Detection.query.update(dict(is_matched=True, is_checked=False))
    reannotation.models.BboxSample.query.update(dict(is_matched=True))

    total = 0

    for brand in brands:

        print('processing brand {}'.format(brand.name))

        for item in app.db.session.query(reannotation.models.Detection.sample_id).distinct():

            sample_id = item[0]

            im_detections_fp = reannotation.models.Detection.query.filter(and_(reannotation.models.Detection.sample_id == sample_id,
                                                                               reannotation.models.Detection.brand_id == brand.id, reannotation.models.Detection.score > conf_level_fp)).all()


            # im_detections_fn = reannotation.models.Detection.query.filter(and_(reannotation.models.Detection.sample_id==sample_id,
            #     reannotation.models.Detection.brand_id==brand.id,reannotation.models.Detection.score>conf_level_fn)).all()


            gt_bboxes = reannotation.models.BboxSample.query.filter(and_(reannotation.models.BboxSample.sample_id == sample_id,
                                                                         or_(reannotation.models.BboxSample.brand_id == brand.id, reannotation.models.BboxSample.ignore == True))).all()

            # im_detections = [det.serialize() for det in im_detections]
            # gt_bboxes = [b.serialize() for b in gt_bboxes]

            # intersect
            fp_res = compareSimple(gt_bboxes, im_detections_fp)
            # fn_res = compareSimple(gt_bboxes, im_detections_fn)

            # has error
            has_error = False
            if len(fp_res['false_res']) > 0:

                for b in fp_res['false_res']:
                    b.is_matched = False

                has_error = True

            # if len(fn_res['false_gt']) > 0:
            #     for b in fn_res['false_gt']:
            #         b.is_matched = False
            #     has_error = True

            if has_error:
                total += 1

    print('total number of images with error: {}'.format(total))

def query_yes_no(question, default="no"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def get_number(str_value):
    digits_only = filter(lambda x: x.isdigit(), str_value)
    digits_only = ''.join(digits_only)
    number = int(digits_only)
    return number

def compareSimple(gt, result):

    IoUMain = 0.3
    IoUAdjust = 0.3
    IoUIgnore = 0.2

    IoUThreshold = IoUMain
    PointsNumber = 1000

    gt_faces = [f for f in gt if not f.ignore]
    gt_faces_ignore = [f for f in gt if f.ignore]
    res_faces = []
    if len(result) > 0:
        res_faces = sorted(result, key=attrgetter('score'), reverse=True)

    # check with not-ignored faces
    matched_gt = []
    matched_res = []
    ignored_res = []
    ignored_gt = gt_faces_ignore
    false_res = []
    false_gt = []

    fp_faces = []
    for res_face in res_faces:
        if len(gt_faces) <= 0:
            fp_faces.append(res_face)
            continue

        ms = [IoU(res_face, gt_face) for gt_face in gt_faces]
        index, element = max(enumerate(ms), key=itemgetter(1))
        if element >= IoUThreshold:
            matched_gt.append(gt_faces[index])
            matched_res.append(res_face)
            del gt_faces[index]
        else:
            fp_faces.append(res_face)
    false_gt = gt_faces

    # check with ignored faces
    for fp_face in fp_faces:
        if len(gt_faces_ignore) <= 0:
            false_res.append(fp_face)
        else:
            ms = [IoU(fp_face, gt_face) for gt_face in gt_faces_ignore]
            index, element = max(enumerate(ms), key=itemgetter(1))
            if element < IoUIgnore:
                false_res.append(fp_face)
            else:
                ignored_res.append(fp_face)

    ans = { 'matched_gt': matched_gt,
            'matched_res': matched_res,
            'ignored_res': ignored_res,
            'ignored_gt': ignored_gt,
            'false_res': false_res,
            'false_gt': false_gt}

    return ans

# helper functions to calculate intersection-over-union
def overlap(l, r):
    # format: l = [point, length], same for r
    # returns overlap for 1d segments
    if l[0] > r[0]:
        l, r = r, l
    far_r, far_l = map(sum, (r, l))
    if r[0] > far_l:
        return 0
    if far_l > far_r:
        return r[1]
    return far_l - r[0]

def IoU(a, b):

    if hasattr(a, 'x'):
        a_x, a_y, a_w, a_h = a.x, a.y, a.w, a.h
        b_x, b_y, b_w, b_h = b.x, b.y, b.w, b.h
    else:
        a_x, a_y, a_w, a_h = a['x'], a['y'], a['w'], a['h']
        b_x, b_y, b_w, b_h = b['x'], b['y'], b['w'], b['h']

    # format: a = [x, y, w, h], same for b
    # returns overlap percentage: IoverU
    x_overlap = overlap((a_x, a_w), (b_x, b_w))
    y_overlap = overlap((a_y, a_h), (b_y, b_h))
    common = 1. * x_overlap * y_overlap
    union = a_w * a_h + b_w * b_h - common
    return 1. * common / union

def parseBBox(frame):
    id = frame.get("ID")

    ignored = frame.get("Occluded") == "Occluded"
    bbox_id = int(frame.get("ID"))
    x = int(frame.find("X").text)
    y = int(frame.find("Y").text)
    w = int(frame.find("W").text)
    h = int(frame.find("H").text)
    return (bbox_id, create_marked_face(x, y, w, h, ignored))

def create_marked_face(x_, y_, w_, h_, ignore_):
    res = {}
    res['x'] = x_
    res['y'] = y_
    res['w'] = w_
    res['h'] = h_
    res['ignore'] = ignore_
    return res

def load_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    faces_ids = dict()
    marking = dict()

    for child in root:
        if (child.tag == "Photo"):
            f = child.find('File').text
            faces_data = []
            if (child.find('BBoxes') != None):
                for c in child.find('BBoxes'):
                    (bbox_id, cur_face) = parseBBox(c)
                    # unique ids
                    if bbox_id in faces_ids:
                        # skip
                        continue
                    else:
                        faces_ids[bbox_id] = True
                    faces_data.append(cur_face)
            # if len(faces_data) > 0:
            marking[f] = faces_data
    return marking

def bboxes_sim(bboxes1, bboxes2):
    MIN_SCORE = 0.5
    IOU_THRES = 0.3

    bboxes1 = [bbox for bbox in bboxes1 if bbox['score'] > MIN_SCORE]
    bboxes2 = [bbox for bbox in bboxes2 if bbox['score'] > MIN_SCORE]

    if len(bboxes2) > 0:
        bboxes2 = sorted(bboxes2, key=itemgetter('score'), reverse=True)

    false_bboxes2 = []
    matched = []
    for bbox2 in bboxes2:
        if len(bboxes1) <= 0:
            false_bboxes2.append(bbox2)
            continue

        ms = [IoU(bbox2, bbox1) for bbox1 in bboxes1]
        index, element = max(enumerate(ms), key=itemgetter(1))
        if element >= IOU_THRES:
            # matched_gt.append(bboxes1[index])
            # matched_res.append(bbox2)
            matched.append((bboxes1[index], bbox2, element))
            del bboxes1[index]
        else:
            false_bboxes2.append(bbox2)
    false_bboxes1 = bboxes1

    # compute score

    score_pos = 0
    for match_item in matched:
        score_pos += max(match_item[0]['score'], match_item[1]['score']) * 2 * match_item[2]

    score_neg = 0
    for false_bbox1 in false_bboxes1:
        score_neg -= false_bbox1['score'] * IOU_THRES

    for false_bbox2 in false_bboxes2:
        score_neg -= false_bbox2['score'] * IOU_THRES

    return score_pos + score_neg

def filehash(filename):
    return hashlib.md5(open(filename, 'rb').read()).hexdigest()

def remove_simular(imgs_dir, marking):

    SSIM_TRES = 0.2
    NEW_SHAPE = (256, 256)

    imgs_names = list_images(imgs_dir)
    n_images = len(imgs_names)
    imgs_removed = np.zeros(n_images, dtype=bool)

    print('number of images: {}'.format(n_images))

    output_marking = {}
    output_dub_count = {}
    for i in range(0, n_images):
        if imgs_removed[i]:
            continue

        imname1 = imgs_names[i]
        img1_path = os.path.join(imgs_dir, imname1)

        if imname1 not in marking:
            imgs_removed[i] = True
            print('image {}: skip'.format(i))
            continue

        im_bboxes1 = marking[imname1]
        print('image {}:'.format(i))

        output_marking[imname1] = marking[imname1]
        output_dub_count[imname1] = 0

        for j in range(i+1, n_images):
            if imgs_removed[j]:
                continue
            imname2 = imgs_names[j]
            img2_path = os.path.join(imgs_dir, imname2)

            if imname2 not in marking:
                continue

            im_bboxes2 = marking[imname2]
            ssim_value = bboxes_sim(im_bboxes1, im_bboxes2)
            print('image [{}/{}]: {}'.format(i, j, ssim_value))

            if ssim_value > SSIM_TRES:
                # remove file
                imgs_removed[j] = True
                output_dub_count[imname1] += 1

    return output_marking, output_dub_count