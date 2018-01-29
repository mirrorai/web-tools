#!/usr/bin/env bash

python manage.py add_imdb -i '/home/ubuntu/projects/data/common_db/BigSample_240x320/samples_gender.txt' -n 'BigSample_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_annotated.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_test.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_k_fold_0.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --k-fold 0

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_k_fold_1.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --k-fold 1

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_k_fold_2.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --k-fold 2

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_k_fold_3.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --k-fold 3