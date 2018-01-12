# -*- coding:utf-8 -*-

import os
import logging
import textwrap
from datetime import timedelta
from celery.schedules import crontab

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

    # Mail sending
    MAIL_NO_REPLY_SENDER = '{} < no-reply-{}@mirror-ai.com >'.format(SITE_NAME, SITE_NAME.lower())
    MAIL_DEFAULT_SENDER = MAIL_NO_REPLY_SENDER
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

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
    CELERY_BROKER_URL = 'amqp://localhost'
    CELERY_RESULT_BACKEND = 'amqp://localhost'
    CELERY_BACKEND_URL = 'amqp://localhost'
    CELERY_TIMEZONE = 'UTC'

    CELERYBEAT_SCHEDULE = {
        'clean_images_every_5_minutes': {
            'task': 'webtools.cron_tasks.clean_images',
            'schedule': timedelta(seconds=10)
        }
    }


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

    # SqlAlchemy
    # SQLALCHEMY_DATABASE_URI = 'mysql://root:Mug0aenoPhi5koh4@localhost/medicine_dev'
    SQLALCHEMY_DATABASE_URI = 'sqlite:////Users/denemmy/projects/mirror_ai/web-server/web-tools/database/webtools.db'
    SQLALCHEMY_DATABASE_URI = 'sqlite:////home/ubuntu/projects/web-tools/database/webtools.db'

class DevelopmentLocalConfig(Config):
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

    # SqlAlchemy
    SQLALCHEMY_DATABASE_URI = 'sqlite:////Users/denemmy/projects/mirror_ai/web-server/web-tools/database/webtools.db'

class StagingConfig(Config):
    CONFIG_TAG = ('[S]', 'Staging')

    # Environment
    STAGING = True
    SERVER_NAME = 'localhost'

    # Image handling
    IMAGE_FOLDER = '__NOT_SET__'
    IMAGE_CACHE_FOLDER = '__NOT_SET__'

    # SqlAlchemy
    MYSQL_LOGIN = os.environ.get('MYSQL_LOGIN')
    MYSQL_PASS = os.environ.get('MYSQL_PASS')
    SQLALCHEMY_DATABASE_URI = 'mysql://{}:{}@localhost/webtools_staging'.format(MYSQL_LOGIN, MYSQL_PASS)

class ProductionConfig(Config):
    # Environment
    PRODUCTION = True
    SERVER_NAME = 'webtools.mirror-ai.com'

    # Logging
    LOG_LEVEL = logging.INFO

    # Image handling
    IMAGE_FOLDER = '__NOT_SET__'
    IMAGE_CACHE_FOLDER = '__NOT_SET__'

    # SqlAlchemy
    MYSQL_LOGIN = os.environ.get('MYSQL_LOGIN')
    MYSQL_PASS = os.environ.get('MYSQL_PASS')
    SQLALCHEMY_DATABASE_URI = 'mysql://{}:{}@localhost/webtools'.format(MYSQL_LOGIN, MYSQL_PASS)

    # Queue agent
    QUEUE_AGENT_DEFAULT_BRANCH = 'production'


# Run this always even when used as module

# Create image directories
_config_class = eval(os.environ['STAGE_CONFIG'].split('.')[-1])
if not os.path.exists(_config_class.IMAGE_FOLDER):
    os.mkdir(_config_class.IMAGE_FOLDER)
if not os.path.exists(_config_class.IMAGE_CACHE_FOLDER):
    os.mkdir(_config_class.IMAGE_CACHE_FOLDER)