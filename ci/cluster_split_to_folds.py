from os import mkdir, listdir, makedirs
from os.path import join, isdir, exists, isfile, splitext, basename
from shutil import copyfile
import numpy as np
import random
from matplotlib import pyplot as plt

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

def write_results(output_path, person_ids, persons):
    with open(output_path, 'w') as fp:
        for person_id in person_ids:
            for imname in persons[person_id]:
                fp.write('{}\n'.format(imname))

def split_to_folds():
    base_dir = '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320'
    base_dir = '/Users/denemmy/projects/mirror_ai/data/common_db/person_cluster_15_11_17_240x320'
    samples = load_samples(join(base_dir, 'samples_gender_not_annotated.txt'))
    persons = {}
    for imname in samples:
        person_id = int(imname.split('_')[0])
        if person_id not in persons:
            persons[person_id] = []
        persons[person_id].append(imname)

    k_folds = 4
    k_folds_and_test = k_folds + 1
    persons_ids = np.array(persons.keys())
    # ridx = np.arange(len(persons_ids))
    np.random.shuffle(persons_ids)
    # persons_ids = persons_ids[ridx]

    samples_bins = np.array([len(persons[person_id]) for person_id in persons_ids])
    samples_cnt_cumsum = np.cumsum(samples_bins)
    top_value = samples_cnt_cumsum[-1]

    samples_for_fold = int(float(top_value) / k_folds_and_test)
    rest = top_value - samples_for_fold * k_folds_and_test
    cnt_k_folds = np.array([samples_for_fold for i in range(k_folds_and_test)])
    for i in range(rest):
        cnt_k_folds[i] += 1
    np.random.shuffle(cnt_k_folds)
    print(cnt_k_folds)
    offs = 0
    vidx = []
    for i in range(k_folds_and_test-1):
        offs += cnt_k_folds[i]
        v = np.searchsorted(samples_cnt_cumsum, offs)
        vidx.append(v)

    persons_k_folds = {}
    persons_k_folds[0] = persons_ids[:vidx[0]]
    persons_k_folds[k_folds_and_test-1] = persons_ids[vidx[k_folds_and_test-2]:]
    for i in range(1, k_folds_and_test-1):
        persons_k_folds[i] = persons_ids[vidx[i-1]:vidx[i]]

    output_name = 'samples_gender_not_annotated_split_test.txt'
    write_results(join(base_dir, output_name), persons_k_folds[0], persons)
    for i in range(1, k_folds_and_test):
        output_name = 'samples_gender_not_annotated_split_k_fold_{}.txt'.format(i-1)
        write_results(join(base_dir, output_name), persons_k_folds[i], persons)

    plt.plot(samples_cnt_cumsum)
    plt.show()

if __name__ == '__main__':
    split_to_folds()