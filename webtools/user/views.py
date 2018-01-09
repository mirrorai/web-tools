# -*- coding: utf-8 -*-
from flask import render_template
from flask_login import login_required, current_user
from flask_principal import RoleNeed, Permission
from flask_security.forms import ChangePasswordForm
from werkzeug.exceptions import abort

from webtools import app

from .models import User


@app.context_processor
def define_template_globals():
    return dict(
        current_user=current_user,
        RoleNeed=RoleNeed,
        Permission=Permission
    )


@app.route('/user/<int:id>', methods=['GET'])
@login_required
def user(id):
    u = User.query.get_or_404(id)

    # TODO: check organization permissions here in future
    if not current_user.has_role('admin') and u.id != current_user.id:
        abort(403)

    return render_template('user.html', user=u, change_password_form=ChangePasswordForm())


@app.route('/user/my_profile', methods=['GET'])
@login_required
def user_my_profile():
    return user(current_user.id)
