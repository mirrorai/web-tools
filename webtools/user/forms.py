# -*- coding: utf-8 -*-
from flask_login import current_user
from flask_wtf import Form
from flask_security import RegisterForm, ConfirmRegisterForm
from flask_security.forms import Required
from wtforms import StringField, ValidationError
from wtforms.fields.html5 import EmailField
from wtforms.validators import DataRequired

from .models import User


class ExtendedRegisterForm(RegisterForm):
    first_name = StringField('First Name', [Required()])
    middle_name = StringField('Middle Name', [])
    last_name = StringField('Last Name', [Required()])


class ExtendedConfirmRegisterForm(ConfirmRegisterForm):
    first_name = StringField('First Name', [Required()])
    middle_name = StringField('Middle Name', [])
    last_name = StringField('Last Name', [Required()])


class UserReferenceForm(Form):
    email = EmailField('email', validators=[DataRequired()])

    def validate_email(self, field):
        if User.query.filter(User.email == field.data).one_or_none() is None:
            raise ValidationError("User with specified email doesn't exists")


class GrantAccessUserReferenceForm(UserReferenceForm):
    def validate_email(self, field):
        if current_user.email == field.data:
            raise ValidationError("User isn't allowed to edit his own rights")
        super(GrantAccessUserReferenceForm, self).validate_email(field)
