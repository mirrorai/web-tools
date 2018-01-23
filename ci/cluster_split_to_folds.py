from os import mkdir, listdir, makedirs
from os.path import join, isdir, exists, isfile, splitext, basename
from shutil import copyfile
import numpy as np
import random
import cv2

def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list

def load_data(samples_file):

    with open(samples_file) as fp:
        content = fp.read().splitlines()

    data = []
    for line in content:
        parts = line.split(';')
        imname = parts[0]
        label_str = parts[1].lower()
        assert(label_str in ['m', 'f'])
        is_male = label_str == 'm'
        data.append((imname, is_male))

    return data

def load_samples(samples_file):

    with open(samples_file) as fp:
        content = fp.read().splitlines()

    data = []
    for line in content:
        parts = line.split(';')
        imname = parts[0]
        data.append(imname)

    return data

def split_to_folds()