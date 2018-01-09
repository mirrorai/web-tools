# -*- coding:utf-8 -*-
from collections import namedtuple
from functools import partial

from flask import jsonify
from flask_principal import Permission, RoleNeed, UserNeed, identity_loaded
from flask_security import current_user, login_required
from sqlalchemy.orm import class_mapper

from webtools import app
from webtools.utils import camelcase_to_snakecase

from .forms import GrantAccessUserReferenceForm
from .models import User

# Shortcuts
db = app.db

# Permission helpers
identity_setters = []  # Here create_permission_to_resource_instance function registers permission setters to identity


@identity_loaded.connect_via(app)
def on_identity_loaded(_, identity):
    identity.user = current_user

    if hasattr(current_user, 'id'):
        identity.provides.add(UserNeed(current_user.id))

    for role in current_user.roles:
        identity.provides.add(RoleNeed(role.name))

    for identity_setter in identity_setters:
        identity_setter(identity, current_user)


def create_permission_to_resource_instance(resource_type, method):
    # Convert resource_type class name to CamelCase and under_score variants
    resource_name = resource_type.__name__
    resource_name_underscore = camelcase_to_snakecase(resource_name)

    # Create linking table
    two_side_table = db.Table(
        'perm_{}_{}'.format(resource_name_underscore, method),
        db.SDColumn(
            'u_id',
            db.Integer,
            db.ForeignKey('user.id', onupdate='CASCADE', ondelete='CASCADE')
        ),
        db.SDColumn(
            'r_id'.format(resource_name_underscore),
            db.Integer,
            db.ForeignKey(
                '{}.id'.format(resource_name_underscore),
                onupdate='CASCADE',
                ondelete='CASCADE'
            )
        )
    )

    # Set link from user to resources
    class_mapper(User).add_properties({
        '{}s_{}'.format(resource_name_underscore, method):
            db.relationship(resource_type, secondary=two_side_table, uselist=True)
    })

    # Set link from resource to users
    class_mapper(resource_type).add_properties({
        'users_{}'.format(method):
            db.relationship(User, secondary=two_side_table, uselist=True)
    })

    # Need template
    result_need = partial(
        namedtuple(
            '{}{}Need'.format(resource_name, method.capitalize()),
            ['resource', 'method', '{}_id'.format(resource_name_underscore)]
        ),
        resource_name_underscore,
        method
    )

    # Populate identity right way by providing identity setter
    def identity_setter(identity, user):
        for resource in getattr(user, '{}s_{}'.format(resource_name_underscore, method), []):
            identity.provides.add(result_need(resource.id))

    identity_setters.append(identity_setter)

    # Create base
    def get_base_query():
        if current_user.has_role('admin'):
            return resource_type.query
        else:
            return resource_type.query.filter(
                getattr(resource_type, 'users_{}'.format(method)).any(
                    id=current_user.id
                )
            )

    return result_need, get_base_query


def create_permission_to_resource_type(resource_type, method):
    # Convert resource_type class name to CamelCase and under_score variants
    resource_name = resource_type.__name__
    resource_name_underscore = camelcase_to_snakecase(resource_name)

    # Specify can user access type or not
    column = db.SDColumn('{}s_{}'.format(resource_name_underscore, method), db.Boolean, default=False)
    class_mapper(User).mapped_table.append_column(column)
    class_mapper(User).add_properties({
        '{}s_{}'.format(resource_name_underscore, method): column
    })

    # Need template
    result_need = partial(
        namedtuple(
            '{}{}Need'.format(resource_name, method.capitalize()),
            ['resource', 'method']
        ),
        resource_name_underscore,
        method
    )

    # Populate identity right way by providing identity setter
    def identity_setter(identity, user):
        if getattr(user, '{}s_{}'.format(resource_name_underscore, method), False):
            identity.provides.add(result_need())

    identity_setters.append(identity_setter)

    return result_need


def generate_permission_manipulating_endpoints(resource_type, resource_manage_need):
    resource_name_underscore = camelcase_to_snakecase(resource_type.__name__)

    def checked_append(target_list, item):
        if item not in target_list:
            target_list.append(item)

    def checked_remove(target_list, item):
        if item in target_list:
            target_list.remove(item)

    def generate(mode, method):
        def endpoint_builder(id):
            resource = resource_type.query.get_or_404(id)
            Permission(resource_manage_need(id), RoleNeed('admin')).test(403)

            form = GrantAccessUserReferenceForm()
            if form.validate_on_submit():
                user = User.query.filter(User.email == form.email.data).one_or_none()

                all_methods = ['read', 'update', 'manage']

                if method not in all_methods:
                    raise ValueError('`method` should be one of: {}'.format(str(all_methods)))
                method_id = all_methods.index(method)

                if mode == 'grant':
                    for grant_method in all_methods[:(method_id + 1)]:
                        checked_append(
                            getattr(resource, 'users_{}'.format(grant_method)),
                            user
                        )
                elif mode == 'forbid':
                    for forbid_method in all_methods[method_id:]:
                        checked_remove(
                            getattr(resource, 'users_{}'.format(forbid_method)),
                            user
                        )
                else:
                    raise ValueError("'mode' should be one of: ['grant', 'forbid']")

                db.session.commit()

                return jsonify({}), 200

            return jsonify(form.errors), 400

        result = endpoint_builder
        result.__name__ = '{}_{}_{}'.format(resource_name_underscore, mode, method)

        return app.route('/{}/<int:id>/{}/{}'.format(resource_name_underscore, mode, method), methods=['POST'])(
            login_required(result)
        )

    return generate('grant', 'read'), generate('grant', 'update'), generate('grant', 'manage'),\
        generate('forbid', 'read'), generate('forbid', 'update'), generate('forbid', 'manage')
