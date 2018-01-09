# -*- coding: utf-8 -*-
from flask_security import RoleMixin, UserMixin
from sqlalchemy_utils import IPAddressType

from webtools import app

# Shortcuts
db = app.db


# Role -- user permission mapping
roles_users = db.Table(
    'principal_role_user',
    db.SDColumn('user_id', db.Integer, db.ForeignKey('user.id')),
    db.SDColumn('role_id', db.Integer, db.ForeignKey('role.id'))
)


class Role(db.Model, RoleMixin):
    id = db.SDColumn(db.Integer, primary_key=True)

    # Role information
    name = db.SDColumn(db.String(128), unique=True)
    description = db.SDColumn(db.String(256))

    # Users with role
    users = db.relationship('User', secondary=roles_users, uselist=True, lazy='dynamic', back_populates='roles')


class User(db.Model, UserMixin):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)

    # User information (This is Flask-Security managed)
    email = db.SDColumn(db.String(128), unique=True, index=True)
    password = db.SDColumn(db.String(256))
    active = db.SDColumn(db.Boolean)
    # FIXME: Change this FlaskSecurity tracked fields below to ArrowType when PR below accepted:
    # https://github.com/mattupstate/flask-security/pull/483
    confirmed_at = db.SDColumn(db.DateTime, nullable=True)
    last_login_at = db.SDColumn(db.DateTime, nullable=True)
    current_login_at = db.SDColumn(db.DateTime, nullable=True)
    last_login_ip = db.SDColumn(IPAddressType, nullable=True)
    current_login_ip = db.SDColumn(IPAddressType, nullable=True)
    login_count = db.SDColumn(db.Integer, default=0)

    first_name = db.SDColumn(db.String(256))
    last_name = db.SDColumn(db.String(256))
    middle_name = db.SDColumn(db.String(256), nullable=True)

    # Access
    roles = db.relationship('Role', secondary=roles_users, uselist=True, back_populates='users')
    # Don't forget that a bunch of permission backrefs is created in permissions.py

    def __init__(self, *args, **kwargs):

        # noinspection PyArgumentList
        super(User, self).__init__(*args, **kwargs)

        if not self.has_role('demo'):
            # Allow creating cameras and triggers if user isn't `demo`
            # self.cameras_create = True
            pass

    def __repr__(self):
        return '<User id: {}, email: {}, name: {} {}>'.format(self.id, self.email, self.first_name, self.last_name)
