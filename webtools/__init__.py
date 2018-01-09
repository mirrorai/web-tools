# -*- coding:utf-8 -*-

# Refer to this boilerplate for project structure:
# https://github.com/hansonkd/FlaskBootstrapSecurity

import logging

import os

flask_app_dir = os.path.dirname(os.path.abspath(__file__))

# Creating Flask application instance
# Configuration is controlled through STAGE_CONFIG environment variable
from flask import Flask
app = Flask(
    __name__,
    template_folder=os.path.join(flask_app_dir, '..', 'templates'),
    static_folder=os.path.join(flask_app_dir, '..', 'static')
)
app.config.from_object(os.environ['STAGE_CONFIG'])
app.logger.info('Starting with %s configuration', os.environ['STAGE_CONFIG'])

# Creating database instance
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
# http://flask-sqlalchemy.pocoo.org/2.1/config/#using-custom-metadata-and-naming-conventions
# http://alembic.zzzcomputing.com/en/latest/naming.html
app.db = SQLAlchemy(
    app,
    metadata=MetaData(
        naming_convention={
            "ix": 'ix_%(column_0_label)s',
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(column_0_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s",
            "pk": "pk_%(table_name)s"
        }
    )
)
# Sane Defaults Column
class SDColumn(app.db.Column):
    def __init__(self, *args, **kwargs):
        kwargs['nullable'] = kwargs.get('nullable', False)
        super(SDColumn, self).__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        if key in ['name']:
            if value is not None:
                value = str(value.lstrip('_'))
        super(SDColumn, self).__setattr__(key, value)

app.db.SDColumn = SDColumn

# Creating e-mail manager
from flask_mail import Mail
app.mail = Mail(app)

# Registering jinja extensions
from .momentjs import MomentJS
app.jinja_env.add_extension('jinja2.ext.do')
app.jinja_env.globals['momentjs'] = MomentJS

# Django like debug toolbar
from flask_debugtoolbar import DebugToolbarExtension
app.toolbar = DebugToolbarExtension(app)

# Bootstrap
from flask_bootstrap import Bootstrap, StaticCDN
Bootstrap(app)
app.extensions['bootstrap']['cdns']['bootstrap'] = StaticCDN(static_endpoint='serve_bootstrap')
app.extensions['bootstrap']['cdns']['jquery'] = StaticCDN(static_endpoint='serve_jquery')

# Setup Flask-Security
from flask_security import Security, SQLAlchemyUserDatastore
from .user.models import User, Role
from .user.forms import ExtendedRegisterForm, ExtendedConfirmRegisterForm
app.user_datastore = SQLAlchemyUserDatastore(app.db, User, Role)
app.security = Security(
    app,
    app.user_datastore,
    register_form=ExtendedRegisterForm,
    confirm_register_form=ExtendedConfirmRegisterForm
)

# Setup Flask-Admin
from flask_admin import Admin, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask_security import current_user

class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.has_role('admin')

app.admin = Admin(app, name='Web-Admin', template_mode='bootstrap3', index_view=MyAdminIndexView())

class ProtectedModelView(ModelView):
    def is_accessible(self):
        return current_user.has_role('admin')

# Setup celery
from celery import Celery

def make_celery(application):
    cel = Celery(application.import_name)
    cel.conf.update(application.config)
    task_base = cel.Task

    class ContextTask(task_base):
        abstract = True

        def __call__(self, *args, **kwargs):
            with application.app_context():
                return task_base.__call__(self, *args, **kwargs)

    cel.Task = ContextTask
    return cel
celery = make_celery(app)
app.celery = celery

# Populating views and models from modules
from . import models
from .reannotation import models
from .history import models
from .user import models

from . import views
from .user import views
from .reannotation import views
from .history import views

# Celery tasks
# import cron_tasks
