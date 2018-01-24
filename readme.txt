
# init VAR before start
###########################
# dev config
export STAGE_CONFIG=config.DevelopmentConfig
export STAGE_CONFIG=config.DevelopmentLocalConfig
# or
# production config
export STAGE_CONFIG=config.ProductionConfig

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

###########################

# database
# change models, no worry about data
1. drop database
2. create database (UTF8): CREATE DATABASE medicine CHARACTER SET utf8;
3. python manage.py db init --> creates 'migrations' folders
4. python manage.py db migrate --> creates commit in 'migrations/versions'
5. python manage.py db upgrade --> actually upgrade database (data can be removed)

# change models, data is important
1. python manage.py db migrate --> creates commit in 'migrations/versions'
2. manually adjust scripts in 'migrations/versions'
3. python manage.py db upgrade --> actually upgrade database

# db populate
1. python manage.py db_command -f data/db_populate_users.json
2. python manage.py db_command -f data/db_populate_hyper_categories.json
2. python manage.py db_command -f data/db_populate_categories.json
3. python manage.py db_command -f data/db_populate_cell_types.json
4. python manage.py db_command -f data/db_populate_images_0.json
5. python manage.py db_command -f data/db_populate_images_1.json
6. python manage.py db_command -f data/db_populate_images_2.json
7. python manage.py db_command -f data/db_add_detector.json
8. python manage.py db_command -f data/db_add_detections.json

# prepare for remarking
1. add detections (step 8 in db populate)
2. python manage.py check_detections

# clean waste images
python manage.py clean_waste_images

# clean cache images
python manage.py clean_cache

# get current marking
python manage.py get_marking -o data/marking.json

# get class labels
python manage.py get_class_labels -o data/class_labels.json

# update locale
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

sudo dpkg-reconfigure locales

# install uwsgi plugin
apt-get install uwsgi-plugin-python

# stop mysql
/etc/init.d/mysql stop

# start mysql
/etc/init.d /mysql start

# restart mysql
/etc/init.d/mysql restart

app.db.session.query(Sample).update(dict(batch_id=None))

# clear ports
lsof -i:8000 -t

# run dev server (run_dev.sh)
python manage.py run_dev_server -h 0.0.0.0 -p 8000

# run production (run_wsgi.sh)
uwsgi_python --socket 0.0.0.0:8000 --workers 4 --protocol=https -w wsgi

# bower install
# install npm firstly
npm install -g bower
ln -s /usr/bin/nodejs /usr/bin/node # fix in ubuntu
bower install
bower install --allow-root # fix in ubuntu

# celery
celery -A webtools.celery worker -B --concurrency 1 --loglevel=info
# clear queue
celery -A webtools.celery purge

# rabbitmq
# start
/usr/local/sbin/rabbitmq-server -detached # on background
/usr/local/sbin/rabbitmq-server #

# stop
rabbitmqctl stop

# check no process
ps aux | grep rabbit # --> kill -9 PID
ps aux | grep erlang # --> kill -9 PID
# kill
pkill -f rabbit
pkill -f erlang

rabbitmqctl status
# service rabbitmq-server start
brew services stop rabbitmq
brew services start rabbitmq

brew uninstall rabbitmq
brew install rabbitmq

# celery workers
ps aux|grep 'celery worker'
pkill -f "celery worker"

# autoreload
celery webtools.celery worker -B --concurrency 1 --loglevel=info \
    --autoreload \
    --include=$(find . -name "*.py" -type f | awk '{sub("\./",""); gsub("/", "."); sub(".py",""); print}' ORS=',' | sed 's/.$//')