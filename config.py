# -*- coding:utf-8 -*-

import os
import logging
import textwrap
from datetime import timedelta
from celery.schedules import crontab
from kombu import Queue

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SITE_NAME = 'WebTools'
    CONFIG_TAG = None  # set to (short_tag, long_tag). Short is prepended to page titles, long is shown on navbar

    # CSRF_ENABLED = True
    SECRET_KEY = os.environ.get('SECRET_KEY')
    WTF_CSRF_ENABLED = True

    # Current environment
    DEVELOPMENT = False
    TESTING = False
    STAGING = False
    PRODUCTION = False

    JSON_AS_ASCII = False

    # Debug features
    DEBUG = False
    TEMPLATES_AUTO_RELOAD = False

    # Image handling options
    IMAGE_FOLDER = '__NOT_SET__'
    IMAGE_CACHE_FOLDER = '__NOT_SET__'  # for resized frames
    TEMP_FOLDER = '__NOT_SET__'
    TRAINROOM_FOLDER = os.path.join(basedir, 'trainroom/')
    REPOSITORIES_FOLDER = os.path.join(basedir, 'repositories/')
    GENA_FOLDER_NAME = 'gena'
    LIMIT_IMAGES_LOAD = 0

    CV_PARTITION_FOLDS = 4
    CHECKED_TIMES_MAX = 5
    SAMPLES_MIN_ERROR = 0.5
    NEW_SAMPLES_MIN_ERROR = 0.2
    SEND_EXPIRE_MIN = 4
    KEEP_TOP_MODELS_CNT = 3

    TRIGGER_TRAIN_MIN_SAMPLES = 1000
    TRIGGER_TRAIN_MAX_HOURS = 24
    TRIGGER_TRAIN_MIN_HOURS = 3

    TRIGGER_TEST_MIN_MINUTES = 20

    TRIGGER_TRAIN_K_FOLDS_MIN_SAMPLES = 1000
    TRIGGER_TRAIN_K_FOLDS_MAX_HOURS = 24
    TRIGGER_TRAIN_K_FOLDS_MIN_HOURS = 3

    TRIGGER_TEST_K_FOLDS_MIN_MINUTES = 30

    MIN_ACCURACY_TO_DEPLOY = 0.98
    TRIGGER_DEPLOY_MIN_HOURS = 3

    GPU_IDS = [3]

    # Mail sending
    MAIL_NO_REPLY_SENDER = '{} < no-reply-{}@mirror-ai.com >'.format(SITE_NAME, SITE_NAME.lower())
    MAIL_DEFAULT_SENDER = MAIL_NO_REPLY_SENDER
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

    GIT_LOGIN = os.environ.get('GIT_LOGIN')
    GIT_PASS = os.environ.get('GIT_PASS')
    GENA_GIT = 'https://{}:{}@github.com/mirrorai/gena.git'.format(GIT_LOGIN, GIT_PASS)

    SLACK_API_TOKEN = os.environ.get('SLACK_API_TOKEN')

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # User management
    SECURITY_EMAIL_SENDER = MAIL_NO_REPLY_SENDER
    SECURITY_CONFIRMABLE = True  # Confirm registrations
    SECURITY_REGISTERABLE = True  # Allow registrations
    SECURITY_RECOVERABLE = False  # Allow password recovery
    SECURITY_TRACKABLE = True  # Track login dates and ips
    SECURITY_CHANGEABLE = True  # Allow password change
    SECURITY_PASSWORD_HASH = 'pbkdf2_sha512'
    SECURITY_PASSWORD_SALT = os.environ.get('SECURITY_PASSWORD_SALT')
    SECURITY_TOKEN_AUTHENTICATION_HEADER = 'AuthToken'
    SECURITY_TOKEN_AUTHENTICATION_KEY = 'token'

    # Celery
    RABBITMQ_NAME = os.environ.get('RABBITMQ_NAME')
    RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS')
    RABBITMQ_VHOST = os.environ.get('RABBITMQ_VHOST')

    # CELERY_BROKER_URL = 'amqp://{}:{}@localhost/{}'.format(RABBITMQ_NAME, RABBITMQ_PASS, RABBITMQ_VHOST)
    # CELERY_RESULT_BACKEND = 'amqp://{}:{}@localhost/{}'.format(RABBITMQ_NAME, RABBITMQ_PASS, RABBITMQ_VHOST)

    CELERY_BROKER_URL = 'amqp://guest@localhost:5672//'
    # CELERY_BACKEND_URL = 'amqp://localhost/'
    # CELERY_RESULT_BACKEND = 'amqp://guest:guest@localhost:5672//'
    CELERY_TASK_PROTOCOL = 1

    CELERY_TIMEZONE = 'UTC'

    CELERYBEAT_SCHEDULE = {
        'print_echo_every_N_minutes': {
            'task': 'webtools.cron_tasks.print_echo',
            'schedule': timedelta(minutes=120)
        },
        'clean_waste_models_every_N_hours': {
            'task': 'webtools.cron_tasks.clean_waste_models',
            'schedule': timedelta(hours=1)
        },
        'auto_train_check_every_N_minutes': {
            'task': 'webtools.reannotation.cron_tasks.auto_training',
            'schedule': timedelta(minutes=40)
        },
        'auto_test_check_every_N_minutes': {
            'task': 'webtools.reannotation.cron_tasks.auto_testing',
            'schedule': timedelta(minutes=10)
        },
        'auto_train_k_folds_check_every_N_minutes': {
            'task': 'webtools.reannotation.cron_tasks.auto_training_k_folds',
            'schedule': timedelta(minutes=60)
        },
        'auto_test_k_folds_check_every_N_minutes': {
            'task': 'webtools.reannotation.cron_tasks.auto_testing_k_folds',
            'schedule': timedelta(minutes=10)
        },
        'annotation_statistics': {
            'task': 'webtools.reannotation.cron_tasks.annotation_statistics',
            'schedule': crontab(minute=0, hour=9)
        },
        'auto_deploy_check_every_N_minutes': {
            'task': 'webtools.reannotation.cron_tasks.auto_deploy',
            'schedule': timedelta(minutes=120)
        }
    }

    # CELERY_TASK_ROUTES = {
    #     'webtools.cron_tasks.*': {'queue': 'celery'},
    #     'webtools.reannotation.celery_tasks.*': {'queue': 'learning'}
    # }

class DevelopmentConfig(Config):
    CONFIG_TAG = ('[D]', 'Development')

    # Current environment
    DEVELOPMENT = True
    # SERVER_NAME = 'localhost'

    # Debug features
    DEBUG = True
    DEBUG_TB_INTERCEPT_REDIRECTS = False  # Debug toolbar redirect interception
    TEMPLATES_AUTO_RELOAD = True
    DEBUG_TB_ENABLED = False

    # Image handling
    IMAGE_FOLDER = os.path.join(basedir, 'images/')
    IMAGE_CACHE_FOLDER = os.path.join(basedir, 'cache_images/')
    TEMP_FOLDER = os.path.join(basedir, 'temp_data/')
    LIMIT_IMAGES_LOAD = 2000

    # SqlAlchemy
    MYSQL_PASS = os.environ.get('MYSQL_PASS')
    MYSQL_DBNAME = os.environ.get('MYSQL_DBNAME')
    SQLALCHEMY_DATABASE_URI = 'mysql://root:{}@localhost/{}'.format(MYSQL_PASS, MYSQL_DBNAME)
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database/webtools.db')

class StagingConfig(Config):
    CONFIG_TAG = ('[S]', 'Staging')

    # Environment
    STAGING = True
    SERVER_NAME = 'localhost'

    # Image handling
    IMAGE_FOLDER = '__NOT_SET__'
    IMAGE_CACHE_FOLDER = '__NOT_SET__'

    # SqlAlchemy
    MYSQL_PASS = os.environ.get('MYSQL_PASS')
    MYSQL_DBNAME = os.environ.get('MYSQL_DBNAME')
    SQLALCHEMY_DATABASE_URI = 'mysql://root:{}@localhost/{}'.format(MYSQL_PASS, MYSQL_DBNAME)

class ProductionConfig(Config):
    # Environment
    PRODUCTION = True
    SERVER_NAME = 'train4.mirror-ai.net:8000'

    # Logging
    LOG_LEVEL = logging.INFO

    # Image handling
    IMAGE_FOLDER = os.path.join(basedir, 'images/')
    IMAGE_CACHE_FOLDER = os.path.join(basedir, 'cache_images/')
    TEMP_FOLDER = os.path.join(basedir, 'temp_data/')

    # LIMIT_IMAGES_LOAD = 2000

    # SqlAlchemy
    MYSQL_PASS = os.environ.get('MYSQL_PASS')
    MYSQL_DBNAME = os.environ.get('MYSQL_DBNAME')
    SQLALCHEMY_DATABASE_URI = 'mysql://root:{}@localhost/{}'.format(MYSQL_PASS, MYSQL_DBNAME)

    # Queue agent
    QUEUE_AGENT_DEFAULT_BRANCH = 'production'


# Run this always even when used as module

# Create image directories
_config_class = eval(os.environ['STAGE_CONFIG'].split('.')[-1])
if not os.path.exists(_config_class.IMAGE_FOLDER):
    os.mkdir(_config_class.IMAGE_FOLDER)
if not os.path.exists(_config_class.IMAGE_CACHE_FOLDER):
    os.mkdir(_config_class.IMAGE_CACHE_FOLDER)
