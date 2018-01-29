import os

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def load_samples(input_file):
    with open(input_file) as fp:
        content = fp.read().splitlines()

    samples = []
    skipped = 0
    for line in content:
        parts = line.split(';')
        local_path = parts[0]
        if not is_ascii(local_path):
            skipped += 1
            continue
        if len(parts) == 2:
            label = parts[1]
        else:
            label = -1
        samples.append((local_path, label))

    return samples, skipped

if __name__ == '__main__':


    input_file = '/Users/denemmy/projects/mirror_ai/data/common_db/AgeDB_240x320/samples.txt'
    input_file = '/home/ubuntu/projects/data/common_db/AgeDB_240x320/samples.txt'

    data, skipped = load_samples(input_file)

    print('number of samples: {}, skipped: {}'.format(len(data), skipped))