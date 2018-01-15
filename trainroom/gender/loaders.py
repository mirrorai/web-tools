import json
from os import listdir
from os.path import dirname, join, splitext, isfile
from image_sample import SampleWithCache

def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp', '.ppm']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list

def load_images_from_csv(csv_path, cfg):

    imgs_dir = dirname(csv_path)
    with open(csv_path) as fp:
        content = fp.read().splitlines()

    samples = []
    for line in content:
        parts = line.split(';')
        img_local_path = parts[0]
        label = int(parts[1])
        mask_local_path = parts[2]

        image_path = join(imgs_dir, img_local_path)
        mask_path = join(imgs_dir, mask_local_path)

        image_sample = SampleWithCache(image_path, label, mask_path, cfg)
        samples.append(image_sample)

    return samples

def load_images_from_dir(imgs_dir, cfg):

    imgs_names = list_images(imgs_dir)

    samples = []
    for imname in imgs_names:
        if imname.endswith('_segmentation.png'):
            continue
        mask_name = '{}_segmentation.png'.format(splitext(imname)[0])
        if not isfile(join(imgs_dir, mask_name)):
            continue
        image_path = join(imgs_dir, imname)
        mask_path = join(imgs_dir, mask_name)
        label = 0
        image_sample = SampleWithCache(image_path, label, mask_path, cfg)
        samples.append(image_sample)

    return samples