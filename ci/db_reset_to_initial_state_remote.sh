mkdir -p migrations
mkdir -p migrations/versions
mkdir -p database
rm -f migrations/versions/*.py
rm -f migrations/versions/*.pyc
rm -f database/*.db
python manage.py db migrate
python manage.py db upgrade
python manage.py db_command -f data/db_populate_users.json
python manage.py add_imdb -i '/home/ubuntu/projects/data/common_db/BigSample_240x320/samples_gender.txt' -n 'BigSample_240x320' -t 'gender'
python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_annotated.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1
python manage.py add_imdb \
    -i '/home/ubuntu/projects/data/common_db/person_cluster_15_11_17_240x320/samples_gender_not_annotated_split_test.txt' \
    -n 'person_cluster_15_11_17_240x320' -t 'gender' --test-only 1
