# -*- coding: utf-8 -*-
from flask_security import RoleMixin, UserMixin
from sqlalchemy_utils import IPAddressType

from webtools import app

# Shortcuts
db = app.db
