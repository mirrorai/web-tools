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

def write_res(output_file, samples, write_label=False):
    with open(output_file, 'w') as fp:
        if write_label:
            for s, l in samples:
                fp.write('{};{}\n'.format(s, 'm' if l else 'f'))
        else:
            for s in samples:
                fp.write('{}\n'.format(s))


def expand_annotation():
    base_dir = '/Users/denemmy/projects/mirror_ai/data/common_db'
    # base_dir = '/home/ubuntu/projects/data/common_db'
    userset_input_file = join(base_dir, 'userset_240x320', 'samples_gender.txt')
    userset_data = load_data(userset_input_file)
    userset_data_map = {}
    for sample, label in userset_data:
        userset_data_map[sample] = label

    cluster_dir = join(base_dir, 'person_cluster_15_11_17_240x320')
    cluster_samples = load_samples(join(cluster_dir, 'samples.txt'))

    persons = {}
    for sample in cluster_samples:
        person_id = int(sample.split('_')[0])
        if person_id not in persons:
            persons[person_id] = []
        persons[person_id].append(sample)

    samples_test = []
    samples_other = []
    skipped = 0
    accepted = 0
    for person_id in persons:
        label = None
        for sample in persons[person_id]:
            sample_base, sample_ext = splitext(sample)
            sample_name = '_'.join(sample_base.split('_')[3:-1])
            sample_id = '{}{}'.format(sample_name, sample_ext)
            if sample_id in userset_data_map:
                label = userset_data_map[sample_id]
                break
        if label is not None:
            samples_test.extend(zip(persons[person_id], [label for i in range(len(persons[person_id]))]))
            accepted += 1
        else:
            samples_other.extend(persons[person_id])
            skipped += 1

    output_test_file = join(cluster_dir, 'samples_gender_annotated.txt')
    output_other_file = join(cluster_dir, 'samples_gender_not_annotated.txt')

    print('skipped {} accepted {}'.format(skipped, accepted))
    print('test count {} --> {}'.format(len(userset_data), len(samples_test)))

    write_res(output_test_file, samples_test, write_label=True)
    write_res(output_other_file, samples_other)

if __name__ == '__main__':
    expand_annotation()