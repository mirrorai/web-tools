# -*- coding:utf-8 -*-
from datetime import timedelta

from flask import redirect, url_for

from . import app
from .utils import camelcase_to_snakecase, snakecase_to_camelcase


@app.route('/static/bootstrap/<path:filename>')
def serve_bootstrap(filename):
    return app.send_static_file('bower_components/bootstrap/dist/' + filename)


@app.route('/static/jquery/<path:filename>')
def serve_jquery(filename):
    return app.send_static_file('bower_components/jquery/dist/' + filename)


# @app.route('/login/csrf')
# def login_csrf():
#     form = LoginForm()
#     return jsonify({'csrf_token': form.csrf_token.current_token})


@app.context_processor
def define_template_globals():
    return dict(
        config_tag=app.config['CONFIG_TAG'],
        camelcase_to_snakecase=camelcase_to_snakecase,
        snakecase_to_camelcase=snakecase_to_camelcase,
        timedelta=timedelta
    )

@app.route('/')
def index():
    return redirect(url_for('reannotation'))
