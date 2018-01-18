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
from sqlalchemy import func, or_, and_, desc, not_, update
from sqlalchemy.orm import joinedload

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

from .models import Image, GenderSample, GenderUserAnnotation, LearningTask
from .forms import GenderDataForm
from .celery_tasks import dump_task
from .celery_tasks import run_train, train_on_error, train_on_success, clear_data_for_train_task
from .celery_tasks import run_test, test_on_error, test_on_success, clear_data_for_test_task
import celery

# Shortcuts
db = app.db

def get_trigger_url_for(problem_name, problem_type):

    if problem_type == 'train':
        return url_for('trigger_train', problem_name=problem_name)
    elif problem_type == 'test':
        return url_for('trigger_test', problem_name=problem_name)
    else:
        return '#'

def get_stop_url_for(problem_type, task_id):

    if problem_type == 'train':
        return url_for('stop_train', task_id=task_id)
    elif problem_type == 'test':
        return url_for('stop_test', task_id=task_id)
    else:
        return '#'

def get_problem_types_labels_map():
    labels = {}
    labels['train'] = 'Train'
    labels['test'] = 'Test'
    labels['find_errors'] = 'Find errors'
    labels['deploy'] = 'Deploy'
    return labels

def get_problem_types_order():
    labels = {}
    labels['train'] = 1
    labels['test'] = 2
    labels['find_errors'] = 3
    labels['deploy'] = 4
    return labels

def get_learning_tasks(problem_name):

    learning_task = LearningTask.query.filter_by(problem_name=problem_name).all()
    labels_map = get_problem_types_labels_map()
    order_map = get_problem_types_order()
    resp_map = {}
    for db_task in learning_task:
        item = {}
        item['progress'] = db_task.progress
        item['state'] = db_task.state
        item['status'] = db_task.status
        item['started_ts'] = db_task.started_ts
        item['finished_ts'] = db_task.finished_ts
        item['start_url'] = get_trigger_url_for(problem_name, db_task.problem_type)
        item['stop_url'] = get_stop_url_for(db_task.problem_type, db_task.task_id)
        item['label'] = labels_map[db_task.problem_type]
        item['order'] = order_map[db_task.problem_type]
        resp_map[db_task.problem_type] = item

    problem_types_all = ['train', 'test', 'find_errors', 'deploy']
    for problem_type in problem_types_all:
        if problem_type not in resp_map:
            item = {}
            item['start_url'] = get_trigger_url_for(problem_name, problem_type)
            item['stop_url'] = '#'
            item['label'] = labels_map[problem_type]
            item['order'] = order_map[problem_type]
            resp_map[problem_type] = item
    resp_list = []
    for problem_type in resp_map:
        item = resp_map[problem_type]
        item['problem_type'] = problem_type
        resp_list.append(item)
    resp_list = sorted(resp_list, key = lambda x: x['order'])
    return resp_list

def get_gender_problem_data():
    gender_stats = get_gender_stats()
    tasks = get_learning_tasks('gender')

    gender_problem_data = {'name_id': 'gender',
                           'name': 'Gender',
                           'annotation_url': url_for('reannotation', problem='gender'),
                           'stats': gender_stats,
                           'enabled': True,
                           'tasks': tasks}
    return gender_problem_data

def get_problems():
    problems = []

    problems.append(get_gender_problem_data())
    problems.append({'name_id': 'hair_color', 'name': 'Hair color', 'enabled': False})
    problems.append({'name_id': 'eyes_color', 'name': 'Eyes color', 'enabled': False})
    problems.append({'name_id': 'race', 'name': 'Race', 'enabled': False})

    return problems

def get_gender_stats():

    total = GenderSample.query.count()

    total_checked = GenderSample.query.filter_by(is_checked=True).count()
    user_checked = GenderUserAnnotation.query.filter_by(user_id=current_user.id).count()

    # total_reannotated = GenderUserAnnotation.query.filter(or_(GenderUserAnnotation.is_changed==True,
    #                                                               GenderUserAnnotation.is_bad==True,
    #                                                               GenderUserAnnotation.is_hard==True)).count()
    # user_reannotated = GenderUserAnnotation.query.filter(and_(GenderUserAnnotation.user_id==current_user.id,
    #                                                            or_(GenderUserAnnotation.is_changed==True,
    #                                                                GenderUserAnnotation.is_bad==True,
    #                                                                GenderUserAnnotation.is_hard==True))).count()

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

def gender_problem(request):
    samples, is_male = get_random_gender_samples(request.url_root)
    form = GenderDataForm()
    stats = get_gender_stats()
    ctx = {'stats': stats, 'is_empty': len(samples) == 0, 'samples': samples,
           'is_male': is_male, 'ts': int(time.time())}
    return render_template('reannotation_gender.html', ctx=ctx, form=form)

@app.route('/reannotation', defaults={'problem': None}, methods=['GET'])
@app.route('/reannotation/<string:problem>', methods=['GET'])
@login_required
@nocache
def reannotation(problem):
    problems = get_problems()
    ctx = {'ts': int(time.time())}
    if not problem or problem not in [p['name_id'] for p in problems]:
        return render_template('reannotation.html', update_url=url_for('reannotation_data'), problems=problems, ctx=ctx)
    else:
        if problem == 'gender':
            return gender_problem(request)

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

        # delete previous annotation for this user and samples
        list_of_ids = [sample_id for sample_id in gender_data]
        deleted = GenderUserAnnotation.query.\
            filter(and_(GenderUserAnnotation.sample_id.in_(list_of_ids),
                        GenderUserAnnotation.user_id==user_id)).delete(synchronize_session='fetch')

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
            # create instance
            user_gender_ann = GenderUserAnnotation(**args)

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
        # flash('Failed to process.')
    else:
        updated = deleted
        added = accepted - deleted
        # flash('Data successfully updated: added: {}, updated: {}, deleted: {}'.format(added, updated, deleted))
    return redirect(url_for('reannotation', problem='gender'))

@app.route('/lenadrobik')
@nocache
def lenadrobik():
    return render_template('base_temp.html')

def update_gender_cv_partition():

    samples = GenderSample.query \
        .filter_by(always_test=False,k_fold=None) \
        .order_by(func.random()).with_entities(GenderSample.id)

    n_samples = samples.count()
    if n_samples == 0:
        return

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    # partition number for each fold
    bins = [int(n_samples / k_folds) for i in range(k_folds)]
    rest = n_samples - sum(bins)
    for i in range(rest):
        bins[i] += 1
    random.shuffle(bins)
    for k_fold in range(k_folds):
        cnt = bins[k_fold]
        subset = GenderSample.query \
            .filter_by(always_test=False,k_fold=None) \
            .order_by(func.random()).with_entities(GenderSample.id) \
            .limit(cnt).with_entities(GenderSample.id)
        subset_to_update = GenderSample.query.filter(GenderSample.id.in_(subset))
        subset_to_update.update(dict(k_fold=k_fold), synchronize_session='fetch')

    app.db.session.commit()

@app.route('/trigger_train/<problem_name>')
@login_required
@nocache
def trigger_train(problem_name):
    if problem_name == 'gender':
        task_id = trigger_train_gender()
        if task_id is None:
            return jsonify(status='error', message='failed to start training'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), task_id=task_id, message='training started.'), 202

def trigger_train_gender():
    if check_working_tasks('gender'):
        print('attempted to start training while other task not finished')
        return None

    clear_old_tasks('gender', 'train')

    task_id = celery.uuid()
    utc = arrow.utcnow()
    task_db = LearningTask(problem_name='gender', problem_type='train',
                           task_id=task_id, started_ts=utc)
    app.db.session.add(task_db)
    app.db.session.flush()
    app.db.session.commit()

    task = run_train.apply_async(('gender',), task_id=task_id, link_error=train_on_error.s(), link=train_on_success.s())

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_train/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_train(task_id):
    celery.task.control.revoke(task_id, terminate=True)
    clear_data_for_train_task(task_id)

    print('task {} successfully stopped'.format(task_id))
    response = dict(status='ok', stopped=True, problems=get_problems(),
                    message='task with id={} successfully stopped'.format(task_id))
    return jsonify(response), 202

@app.route('/trigger_test/<problem_name>')
@login_required
@nocache
def trigger_test(problem_name):
    if problem_name == 'gender':
        task_id = trigger_test_gender()
        if task_id is None:
            return jsonify(status='error', message='failed to start testing'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), task_id=task_id, message='testing started.'), 202

def trigger_test_gender():
    if check_working_tasks('gender'):
        print('attempted to start testing while other task not finished')
        return None

    clear_old_tasks('gender', 'test')

    task_id = celery.uuid()
    utc = arrow.utcnow()
    task_db = LearningTask(problem_name='gender',problem_type='test',
                           task_id=task_id,started_ts=utc)
    app.db.session.add(task_db)
    app.db.session.flush()
    app.db.session.commit()

    task = run_test.apply_async(('gender',), task_id=task_id, link_error=test_on_error.s(), link=test_on_success.s())

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_test/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_test(task_id):
    celery.task.control.revoke(task_id, terminate=True)
    clear_data_for_test_task(task_id)

    print('task {} successfully stopped'.format(task_id))
    response = dict(status='ok', stopped=True, problems=get_problems(),
                    message='task with id={} successfully stopped'.format(task_id))
    return jsonify(response), 202

def check_working_tasks(problem_name):
    return LearningTask.query.filter_by(problem_name=problem_name,finished_ts=None).count() > 0

def stop_all_learning_task(problem_name):
    learning_task = LearningTask.query.filter_by(problem_name=problem_name)
    stopped = 0
    for learning_task in learning_task.all():
        task_id = learning_task.task_id
        celery.task.control.revoke(task_id, terminate=True)
        train_on_error(task_id)
        stopped += 1
    print('{} tasks successfully stopped'.format(stopped))
    learning_task.delete()
    app.db.session.flush()
    app.db.session.commit()

def clear_old_tasks(problem_name, problem_type):
    learning_task = LearningTask.query.filter_by(problem_name=problem_name,problem_type=problem_type)
    print('{} tasks successfully deleted'.format(learning_task.count()))
    learning_task.delete()
    app.db.session.flush()
    app.db.session.commit()

@app.route('/image/<int:id>', defaults={'minside': 0, 'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>', defaults={'maxside': 0}, methods=['GET'])
@app.route('/image/<int:id>/<int:minside>/<int:maxside>', methods=['GET'])
@nocache
@login_required
def image(id, minside, maxside):
    image = Image.query.get_or_404(id)
    return image.send_image(minside=minside, maxside=maxside)

@app.route('/task_status/<task_id>')
@login_required
def task_status(task_id):

    task = run_train.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state == 'REVOKED':
        # job finished
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
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

def get_task_status(task_id):
    task = dump_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state == 'REVOKED':
        # job finished
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
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return response

@app.route('/status/')
@login_required
def status():
    current_tasks = LearningTask.query.all()
    response = {}
    for db_task in current_tasks:
        problem_name = db_task.problem_name
        if problem_name not in response:
            response[problem_name] = []
        response[problem_name].append(get_task_status(db_task.task_id))
    return jsonify(response)

@app.route('/reannotation_data/')
@login_required
def reannotation_data():
    response = get_problems()
    return jsonify(response)
