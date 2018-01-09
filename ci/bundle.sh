#!/bin/sh

set -e

cd /home/local/queue/

sudo apt-get install -y \
    python python-opencv python-numpy python-pip python-flask screen python-flask-sqlalchemy python-flask-migrate \
    python-mysqldb npm nodejs-legacy libgeos-c1v5 libmysqlclient-dev python-dev rabbitmq-server
sudo pip install -r requirements.txt
sudo npm install -g bower
sudo bower install --allow-root --config.interactive=false

python manage.py db upgrade

export C_FORCE_ROOT=1
celery -A queue.celery worker -B &
sh run_wsgi.sh
