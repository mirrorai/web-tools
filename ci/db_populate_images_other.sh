#!/usr/bin/env bash

python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/AgeDB_240x320/samples.txt' \
    -n 'AgeDB_240x320' -t 'gender'


#python manage.py add_imdb \
#    -i '/home/ubuntu/projects/data/common_db/AdienceFaces_240x320/samples.txt' \
#    -n 'AdienceFaces_240x320' -t 'gender'
