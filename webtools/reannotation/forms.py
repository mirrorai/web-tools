# -*- coding: utf-8 -*-
from flask_wtf import Form
from wtforms import StringField, FloatField, IntegerField, BooleanField, SelectField, HiddenField
from wtforms.validators import InputRequired, Length, Regexp, NumberRange, Optional
from wtforms.widgets import HiddenInput

from webtools.forms import ShapelyPolygonField, JsonField

class GenderDataForm(Form):
    is_male = IntegerField("Male", widget=HiddenInput())
    gender_data = JsonField("Gender data")


