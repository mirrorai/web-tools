# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import shutil
import tempfile
import urllib
import zipfile
import random
import json
from operator import itemgetter
from furl import furl

import arrow
import os
from datetime import timedelta
from flask import abort, flash, jsonify, render_template, request, send_file, url_for, redirect
from flask_login import current_user, login_required
from flask_principal import Permission, RoleNeed
from flask_security import auth_token_required
from markupsafe import Markup
from sqlalchemy import func, or_, and_, desc, not_
from sqlalchemy.orm import joinedload

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

from .models import Image, GenderSample

# Shortcuts
db = app.db

@app.context_processor
def define_template_globals():
    return dict(
        is_reannotation=1
    )

def get_tasks(base_url):
    tasks = []

    tasks.append({'url': '', 'id': 'gender', 'name': 'gender', 'count': 1, 'enabled': True})
    tasks.append({'url': '', 'id': 'hair_color', 'name': 'hair color', 'count': 0, 'enabled': False})
    tasks.append({'url': '', 'id': 'eyes_color', 'name': 'eyes color', 'count': 0, 'enabled': False})
    tasks.append({'url': '', 'id': 'race', 'name': 'race', 'count': 0, 'enabled': False})

    for task in tasks:
        task['url'] = furl(base_url).add(query_params=dict(task=task['id'])).url

    return tasks

def get_random_gender_samples(base_url):

    is_male = random.randint(0, 1)
    samples = GenderSample.query.filter_by(is_male=is_male).order_by(func.random()).limit(14).all()
    print('number of sampels: {}'.format(len(samples)))
    samples_data = []
    for sample in samples:
        sample_data = {}
        sample_data['is_male'] = sample.is_male
        sample_data['image'] = base_url + 'image/' + str(sample.image.id)
        samples_data.append(sample_data)
    return samples_data, is_male

@app.route('/reannotation')
@login_required
def reannotation():
    task = request.args.get('task', None)
    tasks = get_tasks(request.base_url)
    if not task or task not in [t['id'] for t in tasks]:
        return render_template('reannotation.html', tasks=tasks)
    else:
        if task == 'gender':
            samples, is_male = get_random_gender_samples(request.url_root)
            ctx = {'samples': samples, 'is_male': is_male}
            return render_template('reannotation_gender.html', ctx=ctx)

@app.route('/image/<int:id>', defaults={'minside': 0, 'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>', defaults={'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>/<int:maxside>', methods=['GET'])
@nocache
@login_required
def image(id, minside, maxside):
    # noinspection PyShadowingNames
    image = Image.query.get_or_404(id)
    # Permission(RoleNeed('admin')).test(403)

    return image.send_image(minside=minside, maxside=maxside)

