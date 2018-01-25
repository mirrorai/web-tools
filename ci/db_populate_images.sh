#!/usr/bin/env bash

python manage.py add_imdb -i '/home/ubuntu/projects/data/common_db/BigSample_240x320/samples_gender.txt' -n 'BigSample_240x320' -t 'gender'

#python manage.py add_imdb \
#    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_annotated.txt' \
#    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_test.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1