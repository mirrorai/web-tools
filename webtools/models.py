# -*- coding:utf-8 -*-
from flask_wtf import Form
from wtforms_alchemy import model_form_factory

from . import app


class AlchemyModelForm(model_form_factory(Form, strip_string_fields=True)):
    # noinspection PyMethodParameters
    @classmethod
    def get_session(self):
        return app.db.session
