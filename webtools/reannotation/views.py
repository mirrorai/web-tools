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
from sqlalchemy.sql.expression import case

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

from .models import Image, GenderSample, GenderUserAnnotation, LearningTask, LearnedModel,\
    AccuracyMetric, GenderSampleResult
from .forms import GenderDataForm
from .celery_tasks import dump_task
from .celery_tasks import run_train, train_on_error, train_on_success, clear_data_for_train_task
from .celery_tasks import run_test, test_on_error, test_on_success, clear_data_for_test_task
from .celery_tasks import run_train_k_folds, train_k_folds_on_error, train_k_folds_on_success, \
    clear_data_for_train_k_folds_task
from .celery_tasks import run_test_k_folds, test_k_folds_on_error, test_k_folds_on_success, \
    clear_data_for_test_k_folds_task
import celery

import numpy as np

# Shortcuts
db = app.db

# common functions
def get_trigger_url_for(problem_name, problem_type):

    if problem_type == 'train':
        return url_for('trigger_train', problem_name=problem_name)
    elif problem_type == 'test':
        return url_for('trigger_test', problem_name=problem_name)
    elif problem_type == 'train_k_folds':
        return url_for('trigger_train_k_folds', problem_name=problem_name)
    elif problem_type == 'test_k_folds':
        return url_for('trigger_test_k_folds', problem_name=problem_name)
    else:
        return '#'

def get_stop_url_for(problem_type, task_id):

    if problem_type == 'train':
        return url_for('stop_train', task_id=task_id)
    elif problem_type == 'test':
        return url_for('stop_test', task_id=task_id)
    elif problem_type == 'train_k_folds':
        return url_for('stop_train_k_folds', task_ids_str=task_id)
    elif problem_type == 'test_k_folds':
        return url_for('stop_test_k_folds', task_ids_str=task_id)
    else:
        return '#'

def get_problem_types_labels_map():
    labels = {}
    labels['train'] = 'Train'
    labels['test'] = 'Test'
    labels['train_k_folds'] = 'Train k-folds'
    labels['test_k_folds'] = 'Test k-folds'
    labels['deploy'] = 'Deploy'
    return labels

def get_problem_types_order():
    labels = {}
    labels['train'] = 1
    labels['test'] = 2
    labels['train_k_folds'] = 3
    labels['test_k_folds'] = 4
    labels['deploy'] = 5
    return labels

def get_learning_tasks(problem_name):

    learning_task = LearningTask.query.filter_by(problem_name=problem_name).all()
    labels_map = get_problem_types_labels_map()
    order_map = get_problem_types_order()
    resp_map = {}
    for db_task in learning_task:

        problem_type = db_task.problem_type

        item = {}
        item['progress'] = db_task.progress
        item['state'] = db_task.state
        item['status'] = db_task.status
        item['started_ts'] = db_task.started_ts
        item['finished_ts'] = db_task.finished_ts
        item['task_id'] = db_task.task_id
        item['k_fold'] = db_task.k_fold

        if problem_type not in resp_map:
            problem_type_item = {}
            problem_type_item['start_url'] = get_trigger_url_for(problem_name, problem_type)
            problem_type_item['stop_url'] = get_stop_url_for(problem_type, db_task.task_id)
            problem_type_item['label'] = labels_map[problem_type]
            problem_type_item['order'] = order_map[problem_type]
            problem_type_item['is_finished'] = item['finished_ts'] != None
            problem_type_item['tasks'] = []
            resp_map[problem_type] = problem_type_item

        resp_map[problem_type]['is_finished'] = resp_map[problem_type]['is_finished'] and item['finished_ts'] != None
        resp_map[problem_type]['tasks'].append(item)

    for problem_type in resp_map:
        if len(resp_map[problem_type]['tasks']) > 1:
            task_ids = ','.join([item['task_id'] for item in resp_map[problem_type]['tasks']])
            resp_map[problem_type]['stop_url'] = get_stop_url_for(problem_type, task_ids)
            resp_map[problem_type]['tasks'] = sorted(resp_map[problem_type]['tasks'],key=lambda x: x['k_fold'])

    problem_types_all = ['train', 'test', 'train_k_folds', 'test_k_folds', 'deploy']
    for problem_type in problem_types_all:
        if problem_type not in resp_map:
            item = {}
            item['start_url'] = get_trigger_url_for(problem_name, problem_type)
            item['stop_url'] = '#'
            item['label'] = labels_map[problem_type]
            item['order'] = order_map[problem_type]
            item['is_finished'] = False
            item['tasks'] = []
            resp_map[problem_type] = item

    return resp_map

def task_to_ordered_list(resp_map):
    resp_list = []
    for problem_type in resp_map:
        problem_type_item = resp_map[problem_type]
        problem_type_item['problem_type'] = problem_type
        resp_list.append(problem_type_item)
    resp_list = sorted(resp_list, key=lambda x: x['order'])
    return resp_list

def task_filter(resp_map, models_count, models_k_folds_count, annotated_count):
    if models_count == 0:
        resp_map['test']['start_url'] = '#'
        resp_map['test']['stop_url'] = '#'

        resp_map['deploy']['start_url'] = '#'
        resp_map['deploy']['stop_url'] = '#'

    if models_k_folds_count == 0:
        resp_map['test_k_folds']['start_url'] = '#'
        resp_map['test_k_folds']['stop_url'] = '#'

    if annotated_count == 0:
        resp_map['train']['start_url'] = '#'
        resp_map['train']['stop_url'] = '#'

        resp_map['train_k_folds']['start_url'] = '#'
        resp_map['train_k_folds']['stop_url'] = '#'

def get_learned_models_count(problem_name, k_folds=False):
    if not k_folds:
        models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                                LearnedModel.k_fold==None,
                                                LearnedModel.finished_ts!=None))
    else:
        models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                                LearnedModel.k_fold!=None,
                                                LearnedModel.finished_ts!=None))
    return models.count()

def check_working_tasks(problem_name, problem_type):
    return LearningTask.query.\
               filter_by(problem_type=problem_type,problem_name=problem_name,finished_ts=None)\
               .count() > 0

def stop_all_learning_task(problem_name):
    learning_task = LearningTask.query.filter_by(problem_name=problem_name)
    stopped = 0
    for learning_task in learning_task.all():
        task_id = learning_task.task_id
        celery.task.control.revoke(task_id, terminate=True, queue='learning')
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

def get_problems():
    problems = []

    problems.append(get_gender_problem_data())
    problems.append({'name_id': 'hair_color', 'name': 'Hair color', 'enabled': False})
    problems.append({'name_id': 'eyes_color', 'name': 'Eyes color', 'enabled': False})
    problems.append({'name_id': 'race', 'name': 'Race', 'enabled': False})

    return problems

# gender
def get_gender_problem_data():
    gender_stats = get_gender_stats()
    models_count = get_learned_models_count('gender')
    models_k_folds_count = get_learned_models_count('gender', k_folds=True)
    tasks = get_learning_tasks('gender')
    task_filter(tasks, models_count, models_k_folds_count, gender_stats['total_annotated'])
    tasks = task_to_ordered_list(tasks)

    gender_problem_data = {'name_id': 'gender',
                           'name': 'Gender',
                           'annotation_url': url_for('reannotation', problem='gender'),
                           'stats': gender_stats,
                           'enabled': True,
                           'metrics': get_last_model_gender_metrics(),
                           'tasks': tasks}
    return gender_problem_data

def get_gender_stats():

    total = GenderSample.query.count()

    total_checked = app.db.session.query(GenderSample). \
        filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                    GenderSample.is_bad == False,
                    GenderSample.is_checked==True)). \
        outerjoin(GenderUserAnnotation). \
        filter(or_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
                   and_(GenderUserAnnotation.is_hard == False,
                        GenderUserAnnotation.is_bad == False))).count()

    user_checked = GenderUserAnnotation.query.filter_by(user_id=current_user.id).count()

    total_annotated = app.db.session.query(GenderSample).outerjoin(GenderUserAnnotation).\
                        filter(or_(GenderUserAnnotation.id != None, GenderSample.is_annotated_gt)).count()


    new_min_error = app.config.get('NEW_SAMPLES_MIN_ERROR')
    min_error = app.config.get('SAMPLES_MIN_ERROR')

    to_check = app.db.session.query(GenderSample).\
        outerjoin(GenderUserAnnotation).\
        filter(and_(GenderSample.is_checked==False,
                    GenderSample.is_bad==False,
                    GenderSample.is_hard==False,
                    and_(or_(GenderUserAnnotation.id == None,
                             and_(GenderUserAnnotation.is_hard == False,
                                  GenderUserAnnotation.is_bad == False))),
                    or_(and_(GenderSample.error > min_error,
                             or_(GenderSample.is_annotated_gt,
                                 GenderUserAnnotation.id != None)),
                        and_(GenderSample.error > new_min_error,
                             GenderSample.is_annotated_gt==False,
                             GenderUserAnnotation.id == None)))).count()

    new_samples = app.db.session.query(GenderSample). \
        outerjoin(GenderUserAnnotation). \
        filter(and_(GenderSample.is_checked == False,
                    GenderSample.is_bad == False,
                    GenderSample.is_hard == False,
                    GenderUserAnnotation.id == None,
                    GenderSample.is_annotated_gt == False,
                    GenderSample.error > new_min_error)).count()

    total_reannotated = app.db.session.query(GenderUserAnnotation).count()
    user_reannotated = 0

    stats = {}
    stats['total'] = total
    stats['to_check'] = to_check
    stats['new_samples'] = new_samples
    stats['total_checked'] = total_checked
    stats['user_checked'] = user_checked
    stats['total_reannotated'] = total_reannotated
    stats['user_reannotated'] = user_reannotated
    stats['total_annotated'] = total_annotated

    return stats

def get_gender_metrics():
    problem_name = 'gender'
    models = app.db.session.query(LearnedModel.id,LearnedModel.finished_ts,AccuracyMetric.accuracy).\
        filter(and_(LearnedModel.k_fold==None,
                    LearnedModel.problem_name==problem_name,
                    LearnedModel.finished_ts!=None)).\
        outerjoin(AccuracyMetric).order_by(LearnedModel.id).all()

    metrics_data = {}
    metrics_data['metrics_names'] = ['Accuracy']
    metrics_data['data'] = []
    for m in models:
        item = {}
        item['model_id'] = m.id
        item['finished_ts'] = m.finished_ts.format('YYYY-MM-DD HH:mm:ss')
        item['metrics_values'] = [m.accuracy]
        metrics_data['data'].append(item)

    return metrics_data

def get_last_model_gender_metrics():
    problem_name = 'gender'
    models = app.db.session.query(LearnedModel.id,LearnedModel.finished_ts,AccuracyMetric.accuracy).\
        filter(and_(LearnedModel.k_fold==None,
                    LearnedModel.problem_name==problem_name,
                    LearnedModel.finished_ts!=None)).\
        outerjoin(AccuracyMetric).order_by(desc(LearnedModel.id)).limit(2).all()

    if len(models) == 0:
        return None
    else:
        item = {}
        item['metrics_url'] = url_for('metrics', problem=problem_name)
        cur_accuracy = models[0].accuracy
        if cur_accuracy is None:
            item['tested'] = False
        else:
            item['tested'] = True
            item['accuracy'] = cur_accuracy
            item['error_reduction'] = 0.0
            if len(models) > 1:
                prev_accuracy = models[1].accuracy
                if prev_accuracy is not None:
                    cur_error = 1 - cur_accuracy
                    prev_error = 1 - prev_accuracy
                    reduction = prev_error / cur_error if cur_error > 1e-12 else 1.0
                    item['error_reduction'] = reduction
        return item

def get_samples_for_ann():

    max_checked_count = app.config.get('CHECKED_TIMES_MAX')
    while True:
        # annotated images with high error and no checked before
        # samples = get_err_gender_samples(max_checked_count=0)
        samples, is_male = get_new_gender_samples()
        break
        # new images
        #if len(samples) == 0:
        #     samples = get_new_gender_samples()

        # annotated images and checked before
        # if len(samples) == 0:
        #     samples = get_err_gender_samples(max_checked_count=max_checked_count)

    return samples, is_male

def get_new_gender_samples():
    is_male_global = random.randint(0, 1)
    min_error = app.config.get('NEW_SAMPLES_MIN_ERROR')

    for i in range(2):
        ann = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.error > min_error,
                        GenderSample.is_checked == False,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.is_male == is_male_global)). \
            outerjoin(GenderUserAnnotation). \
            filter(and_(GenderUserAnnotation.id == None,  # no gt and user annotation
                        GenderSample.is_annotated_gt==False)).limit(21).all()

        if len(ann) > 0:
            break
        is_male_global = not is_male_global

    print(len(ann))
    utc = arrow.now()
    samples_data = []
    for sample, is_male in ann:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['error'] = sample.error
        sample_data['error_label'] = 'uncertainty'

        sample.is_send = True
        sample.send_timestamp = utc

        samples_data.append(sample_data)

    app.db.session.flush()
    app.db.session.commit()

    return samples_data, is_male_global

def get_err_gender_samples(max_checked_count=0):

    is_male_global = random.randint(0, 1)

    min_error = app.config.get('SAMPLES_MIN_ERROR')

    ann = []
    for i in range(2):
        ann = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.error > min_error,
                        GenderSample.is_male == is_male_global,
                        GenderSample.checked_times <= max_checked_count,
                        GenderSample.is_checked == False,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False)). \
            outerjoin(GenderUserAnnotation). \
            filter(
            or_(and_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
                    GenderSample.is_annotated_gt),
                and_(GenderUserAnnotation.id != None,
                     GenderUserAnnotation.is_hard == False,
                     GenderUserAnnotation.is_bad == False))).order_by(desc(GenderSample.error)).limit(21).all()

        if len(ann) > 0:
            break
        is_male_global = not is_male_global

    utc = arrow.now()
    samples_data = []
    for sample, is_male in ann:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['error'] = sample.error
        sample_data['error_label'] = 'error'

        sample.is_send = True
        sample.send_timestamp = utc

        samples_data.append(sample_data)

    app.db.session.flush()
    app.db.session.commit()

    return samples_data, is_male_global

def gender_problem(request):
    form = GenderDataForm()
    stats = get_gender_stats()
    samples, is_male = get_samples_for_ann()
    ctx = {'stats': stats, 'is_empty': len(samples) == 0, 'samples': samples,
           'is_male': is_male, 'ts': int(time.time())}
    return render_template('reannotation_gender.html', ctx=ctx, form=form)

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
        args['is_changed'] = data_item['is_changed']
        args['is_male'] = (is_male and (data_item['is_changed'] == 0))
        args['is_male'] = args['is_male'] or ((not is_male) and (data_item['is_changed'] != 0))
        args['is_hard'] = False
        args['is_bad'] = False

        if 'is_hard' in data_item and check_is_int(data_item['is_hard']):
            args['is_hard'] = data_item['is_hard'] != 0
        if 'is_bad' in data_item and check_is_int(data_item['is_bad']):
            args['is_bad'] = data_item['is_bad'] != 0

        out_data[sample_id_int] = args

    return out_data

def gender_filter_only_changed(gender_data):
    out_data = {}
    for sample_id in gender_data:
        gdata = gender_data[sample_id]
        if gdata['is_bad'] or gdata['is_hard'] or gdata['is_changed']:
            out_data[sample_id] = gdata
    return out_data

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

@app.route('/metrics/<string:problem>', methods=['GET'])
@login_required
@nocache
def metrics(problem):
    if problem not in ['gender']:
        abort(404)
    ctx = {'ts': int(time.time())}
    if problem == 'gender':
        metrics = get_gender_metrics()
    return render_template('metrics.html', metrics=metrics, ctx=ctx)

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
        # filt_gender_data = gender_filter_only_changed(gender_data)

        list_of_ids = [sample_id for sample_id in gender_data]

        # select samples data
        samples_data = app.db.session.query(GenderSample, GenderUserAnnotation).filter(GenderSample.id.in_(list_of_ids))\
            .outerjoin(GenderUserAnnotation).filter(GenderUserAnnotation.id==None).all()
        samples_db_data = {sd.GenderSample.id:sd for sd in samples_data}

        if len(samples_db_data) != len(list_of_ids):
            break

        n_changed = 0
        n_not_changed = 0
        for sample_id in list_of_ids:
            gdata = gender_data[sample_id]
            sample_db = samples_db_data[sample_id].GenderSample
            user_ann_db = samples_db_data[sample_id].GenderUserAnnotation

            if user_ann_db is None:
                is_changed = True
                if sample_db.is_annotated_gt:
                    is_changed = False
                    if gdata['is_hard'] != sample_db.is_hard:
                        is_changed = True
                    if gdata['is_bad'] != sample_db.is_bad:
                        is_changed = True
                    if gdata['is_male'] != sample_db.is_male:
                        is_changed = True

                if is_changed:
                    args = {}
                    args['is_hard'] = gdata['is_hard']
                    args['is_bad'] = gdata['is_bad']
                    args['is_male'] = gdata['is_male']
                    args['user_id'] = user_id
                    args['sample_id'] = sample_id
                    args['mark_timestamp'] = utc

                    # create instance
                    user_gender_ann = GenderUserAnnotation(**args)
                    # add to db
                    app.db.session.add(user_gender_ann)
                    app.db.session.flush()
            else:
                is_changed = False
                if gdata['is_hard'] != user_ann_db.is_hard:
                    is_changed = True
                    user_ann_db.is_hard = gdata['is_hard']
                if gdata['is_bad'] != user_ann_db.is_bad:
                    is_changed = True
                    user_ann_db.is_bad = gdata['is_bad']
                if gdata['is_male'] != user_ann_db.is_male:
                    is_changed = True
                    user_ann_db.is_male = gdata['is_male']

                user_ann_db.user_id = user_id
                user_ann_db.mark_timestamp = utc

            if is_changed:
                n_changed += 1
                sample_db.checked_timed = 0
            else:
                n_not_changed += 1
                sample_db.checked_timed += 1
            sample_db.is_checked = True
            sample_db.is_send = False

        print('changed: {}, not changed: {}'.format(n_changed, n_not_changed))

        app.db.session.flush()
        app.db.session.commit()
        failed = False
        break

        # delete previous annotation for this user and samples
        filt_list_of_ids = [sample_id for sample_id in filt_gender_data]
        list_of_ids = [sample_id for sample_id in gender_data]
        deleted = GenderUserAnnotation.query.\
            filter(and_(GenderUserAnnotation.sample_id.in_(filt_list_of_ids),
                        GenderUserAnnotation.user_id==user_id)).delete(synchronize_session='fetch')

        # check that all samples ids is valid
        gender_samples = GenderSample.query.filter(GenderSample.id.in_(list_of_ids))
        if gender_samples.count() != len(list_of_ids):
            break

        for gender_sample in gender_samples.all():
            gender_sample.is_checked = True
            gender_sample.checked_times += 1
            if gender_sample.id in filt_gender_data:
                gender_sample.checked_times = 0

        app.db.session.flush()

        accepted = 0
        for sample_id in filt_gender_data:
            gdata = gender_data[sample_id]

            args = {}
            args['is_hard'] = gdata['is_hard']
            args['is_bad'] = gdata['is_bad']
            args['is_male'] = gdata['is_male']
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
        pass
        # flash('Data successfully updated: added: {}, updated: {}, deleted: {}'.format(added, updated, deleted))
    return redirect(url_for('reannotation', problem='gender'))

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

    if check_working_tasks('gender', 'train'):
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

    task = run_train.apply_async(('gender',), task_id=task_id,
                                 link_error=train_on_error.s(), link=train_on_success.s(),
                                 queue='learning')

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_train/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_train(task_id):
    celery.task.control.revoke(task_id, terminate=True, queue='learning')
    clear_data_for_train_task(task_id, 'REVOKED', 'Stopped')

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
    if check_working_tasks('gender', 'test'):
        print('attempted to start testing while other task not finished')
        return None

    if get_learned_models_count('gender') == 0:
        print('no models for testing')
        return None

    clear_old_tasks('gender', 'test')

    task_id = celery.uuid()
    utc = arrow.utcnow()
    task_db = LearningTask(problem_name='gender',problem_type='test',
                           task_id=task_id,started_ts=utc)
    app.db.session.add(task_db)
    app.db.session.flush()
    app.db.session.commit()

    task = run_test.apply_async(('gender',), task_id=task_id,
                                link_error=test_on_error.s(), link=test_on_success.s(),
                                queue='learning')

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_test/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_test(task_id):
    celery.task.control.revoke(task_id, terminate=True, queue='learning')
    clear_data_for_test_task(task_id, 'REVOKED', 'Stopped')

    print('task {} successfully stopped'.format(task_id))
    response = dict(status='ok', stopped=True, problems=get_problems(),
                    message='task with id={} successfully stopped'.format(task_id))
    return jsonify(response), 202

# train k-folds
@app.route('/trigger_train_k_folds/<problem_name>')
@login_required
@nocache
def trigger_train_k_folds(problem_name):
    if problem_name == 'gender':
        task_ids = trigger_train_k_folds_gender()
        if task_ids is None:
            return jsonify(status='error', message='failed to start training k-folds'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), tasks_id=task_ids, message='training started.'), 202

def trigger_train_k_folds_gender():
    problem_type = 'train_k_folds'
    if check_working_tasks('gender', problem_type):
        print('attempted to start training while other task not finished')
        return None

    clear_old_tasks('gender', problem_type)

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    for k_fold in range(k_folds):
        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name='gender',problem_type=problem_type,
                               task_id=task_id,started_ts=utc,k_fold=k_fold)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_train_k_folds.apply_async(('gender', k_fold), task_id=task_id,
                                             link_error=train_k_folds_on_error.s(),
                                             link=train_k_folds_on_success.s(),
                                             queue='learning')
        task_ids.append(task_id)

    print('{} tasks successfully started'.format(len(task_ids)))
    return task_ids

@app.route('/stop_train_k_folds/<string:task_ids_str>', methods=['GET', 'POST'])
@login_required
def stop_train_k_folds(task_ids_str):

    task_ids = task_ids_str.split(',')
    for task_id in task_ids:
        celery.task.control.revoke(task_id, terminate=True, queue='learning')
        clear_data_for_train_k_folds_task(task_id, 'REVOKED', 'Stopped')

    print('{} tasks successfully stopped'.format(len(task_ids)))
    response = dict(status='ok', stopped=True, problems=get_problems(),
                    message='{} tasks successfully stopped'.format(len(task_ids)))
    return jsonify(response), 202

# test k-folds
@app.route('/trigger_test_k_folds/<problem_name>')
@login_required
@nocache
def trigger_test_k_folds(problem_name):
    if problem_name == 'gender':
        task_ids = trigger_test_k_folds_gender()
        if task_ids is None:
            return jsonify(status='error', message='failed to start training k-folds'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), tasks_id=task_ids, message='training started.'), 202

def trigger_test_k_folds_gender():
    problem_type = 'test_k_folds'
    if check_working_tasks('gender', problem_type):
        print('attempted to start training while other task not finished')
        return None

    clear_old_tasks('gender', problem_type)

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    for k_fold in range(k_folds):
        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name='gender',problem_type=problem_type,
                               task_id=task_id,started_ts=utc,k_fold=k_fold)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_test_k_folds.apply_async(('gender', k_fold), task_id=task_id,
                                            link_error=test_k_folds_on_error.s(),
                                            link=test_k_folds_on_success.s(),
                                            queue='learning')
        task_ids.append(task_id)

    print('{} tasks successfully started'.format(len(task_ids)))
    return task_ids

@app.route('/stop_test_k_folds/<string:task_ids_str>', methods=['GET', 'POST'])
@login_required
def stop_test_k_folds(task_ids_str):

    task_ids = task_ids_str.split(',')
    for task_id in task_ids:
        celery.task.control.revoke(task_id, terminate=True, queue='learning')
        clear_data_for_test_k_folds_task(task_id, 'REVOKED', 'Stopped')

    print('{} tasks successfully stopped'.format(len(task_ids)))
    response = dict(status='ok', stopped=True, problems=get_problems(),
                    message='{} tasks successfully stopped'.format(len(task_ids)))
    return jsonify(response), 202

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
