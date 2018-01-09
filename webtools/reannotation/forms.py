# -*- coding: utf-8 -*-
from flask_wtf import Form
from wtforms import StringField, FloatField, IntegerField, BooleanField, SelectField
from wtforms.validators import InputRequired, Length, Regexp, NumberRange, Optional

from webtools.forms import ShapelyPolygonField



