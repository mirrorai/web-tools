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

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/celebA_aligned/samples_gender.txt' \
    -n 'celebA_aligned' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/tinder_faces/samples_gender.txt' \
    -n 'tinder_faces_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/AgeDB_240x320/samples.txt' \
    -n 'AgeDB_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/AdienceFaces_240x320/samples.txt' \
    -n 'AdienceFaces_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/ChenAge_240x320/samples.txt' \
    -n 'ChenAge_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/tinder_girls_50k/tinder_girls_50k_240x320_clean/samples_gender.txt' \
    -n 'tinder_girls_50k_240x320_clean' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/FGNET2_240x320/samples.txt' \
    -n 'FGNET2_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/RaceRace_240x320/samples_gender.txt' \
    -n 'RaceRace_240x320' -t 'gender'

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/RaceSelfie_240x320/samples_gender.txt' \
    -n 'RaceSelfie_240x320' -t 'gender'
