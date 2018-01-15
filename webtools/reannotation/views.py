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
import time

import arrow
import os
from flask import abort, flash, jsonify, render_template, request, send_file, url_for, redirect
from flask_login import current_user, login_required
from flask_principal import Permission, RoleNeed
from flask_security import auth_token_required
from sqlalchemy import func, or_, and_, desc, not_
from sqlalchemy.orm import joinedload

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

from .models import Image, GenderSample, GenderUserIterAnnotation, GenderIteration
from .forms import GenderDataForm
from celery_tasks import run_train, long_task, train_on_error, train_on_success
import celery

# Shortcuts
db = app.db

@app.context_processor
def define_template_globals():
    return dict(
        is_reannotation=1
    )

def get_gender_task_data():
    gender_stats = get_gender_stats()
    gender_task_data = {'name_id': 'gender', 'name': 'Gender', 'stats': gender_stats, 'enabled': True}
    return gender_task_data

def get_tasks():
    tasks = []

    gender_count = GenderSample.query.filter(GenderSample.is_checked == False).count()
    tasks.append(get_gender_task_data())
    tasks.append({'id': 'hair_color', 'name': 'Hair color', 'stats': None, 'enabled': False})
    tasks.append({'id': 'eyes_color', 'name': 'Eyes color', 'stats': None, 'enabled': False})
    tasks.append({'id': 'race', 'name': 'Race', 'stats': None, 'enabled': False})

    return tasks

def get_gender_stats():

    total = GenderSample.query.count()

    total_checked = GenderSample.query.filter_by(is_checked=True).count()
    user_checked = GenderUserIterAnnotation.query.filter_by(user_id=current_user.id).count()

    # total_reannotated = GenderUserIterAnnotation.query.filter(or_(GenderUserIterAnnotation.is_changed==True,
    #                                                               GenderUserIterAnnotation.is_bad==True,
    #                                                               GenderUserIterAnnotation.is_hard==True)).count()
    # user_reannotated = GenderUserIterAnnotation.query.filter(and_(GenderUserIterAnnotation.user_id==current_user.id,
    #                                                            or_(GenderUserIterAnnotation.is_changed==True,
    #                                                                GenderUserIterAnnotation.is_bad==True,
    #                                                                GenderUserIterAnnotation.is_hard==True))).count()
    total_reannotated = 0
    user_reannotated = 0

    stats = {}
    stats['total'] = total
    stats['to_check'] = total - total_checked
    stats['total_checked'] = total_checked
    stats['user_checked'] = user_checked
    stats['total_reannotated'] = user_reannotated
    stats['user_reannotated'] = total_reannotated

    return stats

def get_random_gender_samples(base_url):

    is_male = random.randint(0, 1)

    # samples = app.db.session.query(GenderSample).\
    #     outerjoin(UserGenderAnnotation, GenderSample.id == UserGenderAnnotation.gender_sample_id).\
    #     filter(UserGenderAnnotation.gender_sample_id==None).order_by(func.random())

    samples = GenderSample.query.filter(GenderSample.is_checked==False)
    samples_gender = samples.filter(GenderSample.is_male==is_male)
    if samples_gender.count() == 0:
        is_male = not is_male
        samples_gender = samples.filter(GenderSample.is_male==is_male)
    samples = samples_gender.order_by(func.random()).limit(21).all()

    samples_data = []
    for sample in samples:
        sample_data = {}
        sample_data['is_male'] = int(sample.is_male)
        sample_data['image'] = base_url + 'image/' + str(sample.image.id)
        sample_data['id'] = sample.id
        samples_data.append(sample_data)
    return samples_data, is_male

def gender_task(request):
    samples, is_male = get_random_gender_samples(request.url_root)
    form = GenderDataForm()
    stats = get_gender_stats()
    ctx = {'stats': stats, 'is_empty': len(samples) == 0, 'samples': samples, 'is_male': is_male, 'ts': int(time.time())}
    return render_template('reannotation_gender.html', ctx=ctx, form=form)

def get_current_iteration(create_if_not_exist=False):
    iterations = GenderIteration.query.order_by(desc(GenderIteration.id))
    if iterations.count() == 0:
        if create_if_not_exist:
            # create iteration
            iteration = create_next_iteration()
        else:
            return None
    else:
        iteration = iterations.first()

    return iteration

def create_next_iteration(is_training=False):
    # create iteration
    iteration = GenderIteration(is_training=is_training)
    app.db.session.add(iteration)
    app.db.session.flush()
    app.db.session.commit()
    return iteration

@app.route('/reannotation', methods=['GET', 'POST'])
@login_required
@nocache
def reannotation():
    task = request.args.get('task', None)
    tasks = get_tasks()
    ctx = {'ts': int(time.time())}
    if not task or task not in [t['id'] for t in tasks]:
        return render_template('reannotation.html', tasks=tasks, ctx=ctx)
    else:
        if task == 'gender':
            return gender_task(request)

def validate_gender_input_data(is_male, gender_data):

    def check_is_int(input_var):
        return isinstance(input_var, int)

    if not check_is_int(is_male):
        return None

    out_data = {}
    for sample_id in gender_data:
        try:
            sample_id_int = int(sample_id)
        except ValueError:
            return None

        data_item = gender_data[sample_id]
        if 'is_changed' not in data_item or not check_is_int(data_item['is_changed']):
            return None
        args = {}
        args['is_male'] = (is_male and (data_item['is_changed'] == 0))
        args['is_male'] = args['is_male'] or ((not is_male) and (data_item['is_changed'] != 0))
        # args['is_changed'] = data_item['is_changed'] != 0

        if 'is_hard' in data_item and check_is_int(data_item['is_hard']):
            args['is_hard'] = data_item['is_hard'] != 0
        if 'is_bad' in data_item and check_is_int(data_item['is_bad']):
            args['is_bad'] = data_item['is_bad'] != 0

        out_data[sample_id_int] = args

    return out_data

@app.route('/update_gender_data', methods=['POST'])
@login_required
@nocache
def update_gender_data():
    form = GenderDataForm(request.form)
    failed = True

    # actually if-statement, break at the end of the loop
    # need for keyword 'break' when checking input data
    while form.validate():

        user_id = current_user.id
        utc = arrow.utcnow()
        is_male = form.is_male.data
        gender_data = form.gender_data.data
        # check json
        gender_data = validate_gender_input_data(is_male, gender_data)
        if gender_data is None:
            break

        iteration = get_current_iteration(create_if_not_exist=True)
        if iteration is None:
            break

        # delete previous annotation for this user, iteration and samples
        list_of_ids = [sample_id for sample_id in gender_data]
        deleted = GenderUserIterAnnotation.query.\
            filter(and_(GenderUserIterAnnotation.sample_id.in_(list_of_ids),
                        GenderUserIterAnnotation.iteration_id==iteration.id,
                        GenderUserIterAnnotation.user_id==user_id)).delete(synchronize_session='fetch')

        # check that all samples ids is valid
        gender_samples = GenderSample.query.filter(GenderSample.id.in_(list_of_ids))
        if gender_samples.count() != len(list_of_ids):
            break
        gender_samples.update(dict(is_checked=True), synchronize_session='fetch')

        accepted = 0
        for sample_id in gender_data:
            args = gender_data[sample_id]
            args['user_id']=user_id
            args['sample_id']=sample_id
            args['mark_timestamp']=utc
            args['iteration_id']=iteration.id
            # create instance
            user_gender_ann = GenderUserIterAnnotation(**args)

            # add to db
            app.db.session.add(user_gender_ann)
            app.db.session.flush()

            accepted += 1

        # emulate if-statement
        failed = False
        app.db.session.commit()
        break

    if failed:
        pass
        flash('Failed to process.')
    else:
        updated = deleted
        added = accepted - deleted
        # flash('Data successfully updated: added: {}, updated: {}, deleted: {}'.format(added, updated, deleted))
    return redirect(url_for('reannotation') + '?task=gender')

@app.route('/lenadrobik')
@nocache
def lenadrobik():
    return render_template('base_temp.html')

@app.route('/trigger_train', methods=['GET', 'POST'])
@login_required
@nocache
def trigger_train():
    data = request.data
    try:
        recv_json = json.loads(data)
    except ValueError:
        return jsonify(status='error', message='wrong post parameters'), 400

    # check parameters
    try:
        task = recv_json['task']
    except KeyError:
        return jsonify(status='error', message='wrong post parameters'), 400

    if task == 'gender':
        status_url, stop_url, task_id = trigger_train_gender()
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', stop_url=stop_url,
                   status_url=status_url, task_id=task_id, message='training started.'), 202

def trigger_train_gender():
    task = run_train.apply_async(('gender',), link_error=train_on_error.s(), link=train_on_success.s())
    return url_for('taskstatus', task_id=task.id), url_for('stop_train', task_id=task.id), task.id

@app.route('/image/<int:id>', defaults={'minside': 0, 'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>', defaults={'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>/<int:maxside>', methods=['GET'])
@nocache
@login_required
def image(id, minside, maxside):
    image = Image.query.get_or_404(id)
    return image.send_image(minside=minside, maxside=maxside)

@app.route('/train_status/<task_id>')
def taskstatus(task_id):
    task = run_train.AsyncResult(task_id)
    print(task.state)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state == 'REVOKED':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': 'Stopped.'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/stop_train/<task_id>', methods=['GET', 'POST'])
def stop_train(task_id):
    celery.task.control.revoke(task_id, terminate=True)
    return jsonify({'status': 'ok', 'stopped': True}), 202

