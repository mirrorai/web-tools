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

from flask.ext.babel import gettext

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

from webtools.user.models import User
from .models import Image, GenderSample, GenderUserAnnotation, LearningTask, LearnedModel,\
    AccuracyMetric, GenderSampleResult, GenderUserAnnotationInfo, GenderUserLog
from .utils import clear_old_tasks, check_working_tasks, get_learned_models_count, get_all_k_folds_learned_models_count

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

admin_permission = Permission(RoleNeed('admin'))
moderator_permission = Permission(RoleNeed('moderator'))

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
            problem_type_item['is_finished'] = item['finished_ts'] != None

            if admin_permission.can():
                problem_type_item['start_url'] = get_trigger_url_for(problem_name, problem_type)
                problem_type_item['stop_url'] = get_stop_url_for(problem_type, db_task.task_id)
            else:
                problem_type_item['start_url'] = '#'
                problem_type_item['stop_url'] = '#'

            problem_type_item['label'] = labels_map[problem_type]
            problem_type_item['order'] = order_map[problem_type]
            problem_type_item['tasks'] = []
            resp_map[problem_type] = problem_type_item

        resp_map[problem_type]['is_finished'] = resp_map[problem_type]['is_finished'] and item['finished_ts'] != None
        resp_map[problem_type]['tasks'].append(item)

    for problem_type in resp_map:
        if len(resp_map[problem_type]['tasks']) > 1:
            task_ids = ','.join([item['task_id'] for item in resp_map[problem_type]['tasks'] if item['finished_ts'] == None])
            if admin_permission.can():
                resp_map[problem_type]['stop_url'] = get_stop_url_for(problem_type, task_ids)
            else:
                resp_map[problem_type]['stop_url'] = '#'
            resp_map[problem_type]['tasks'] = sorted(resp_map[problem_type]['tasks'],key=lambda x: x['k_fold'])

    problem_types_all = ['train', 'test', 'train_k_folds', 'test_k_folds', 'deploy']
    for problem_type in problem_types_all:
        if problem_type not in resp_map:
            item = {}
            if admin_permission.can():
                item['start_url'] = get_trigger_url_for(problem_name, problem_type)
            else:
                item['start_url'] = '#'
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

def get_problems():
    problems = []

    problems.append(get_gender_problem_data())
    # problems.append({'name_id': 'hair_color', 'name': 'Hair color', 'enabled': False})
    # problems.append({'name_id': 'eyes_color', 'name': 'Eyes color', 'enabled': False})
    # problems.append({'name_id': 'race', 'name': 'Race', 'enabled': False})

    return problems

# gender
def get_gender_problem_data():
    gender_stats = get_gender_stats()
    models_count = get_learned_models_count('gender')
    models_k_folds_count = get_all_k_folds_learned_models_count('gender')
    tasks = get_learning_tasks('gender')
    task_filter(tasks, models_count, models_k_folds_count, gender_stats['total_annotated'])
    tasks = task_to_ordered_list(tasks)

    if admin_permission.can():
        gender_problem_data = {'name_id': 'gender',
                               'name': gettext('Gender'),
                               'annotation_url': url_for('reannotation', problem='gender'),
                               'user_control_url': url_for('user_control', problem='gender'),
                               'stats': gender_stats,
                               'enabled': True,
                               'metrics': get_last_model_gender_metrics(),
                               'tasks': tasks}

    elif moderator_permission.can():
        gender_problem_data = {'name_id': 'gender',
                               'name': gettext('Gender'),
                               'annotation_url': url_for('reannotation', problem='gender'),
                               'user_control_url': url_for('user_control', problem='gender'),
                               'stats': gender_stats,
                               'enabled': True,
                               'metrics': get_last_model_gender_metrics()
                               }
    else:
        gender_stats_filt = {}
        gender_stats_filt['total'] = gender_stats['total']
        gender_stats_filt['user_annotated'] = gender_stats['user_annotated']

        gender_problem_data = {'name_id': 'gender',
                               'name': gettext('Gender'),
                               'stats': gender_stats_filt,
                               'annotation_url': url_for('reannotation', problem='gender'),
                               'enabled': True}
    return gender_problem_data

def get_gender_stats():
    utc = arrow.utcnow()
    expected_utc = utc.shift(minutes=-app.config.get('SEND_EXPIRE_MIN'))

    total = GenderSample.query.count()

    # total_checked = app.db.session.query(GenderSample). \
    #     filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
    #                 GenderSample.is_bad == False,
    #                 GenderSample.is_checked==True)). \
    #     outerjoin(GenderUserAnnotation). \
    #     filter(or_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
    #                and_(GenderUserAnnotation.is_hard == False,
    #                     GenderUserAnnotation.is_bad == False))).count()

    # user_checked = GenderUserAnnotation.query.filter_by(user_id=current_user.id).count()

    # total_annotated = app.db.session.query(GenderSample).outerjoin(GenderUserAnnotation).\
    #                     filter(or_(GenderUserAnnotation.id != None, GenderSample.is_annotated_gt)).count()


    new_min_error = app.config.get('NEW_SAMPLES_MIN_ERROR')
    min_error = app.config.get('SAMPLES_MIN_ERROR')

    # to_check = app.db.session.query(GenderSample).\
    #     outerjoin(GenderUserAnnotation).\
    #     filter(and_(GenderSample.is_checked==False,
    #                 GenderSample.is_bad==False,
    #                 GenderSample.is_hard==False,
    #                 or_(GenderSample.send_timestamp == None,
    #                     GenderSample.send_timestamp < expected_utc),
    #                 and_(or_(GenderUserAnnotation.id == None,
    #                          and_(GenderUserAnnotation.is_hard == False,
    #                               GenderUserAnnotation.is_bad == False))),
    #                 or_(and_(GenderSample.error > min_error,
    #                          or_(GenderSample.is_annotated_gt,
    #                              GenderUserAnnotation.id != None)),
    #                     and_(GenderSample.error > new_min_error,
    #                          GenderSample.is_annotated_gt==False,
    #                          GenderUserAnnotation.id == None)))).count()

    # to_check = app.db.session.query(GenderSample).\
    #     outerjoin(GenderUserAnnotation).\
    #     filter(and_(GenderSample.is_checked==False,
    #                 GenderSample.is_bad==False,
    #                 GenderSample.is_hard==False,
    #                 or_(GenderSample.send_timestamp == None,
    #                     GenderSample.send_timestamp < expected_utc),
    #                 and_(or_(GenderUserAnnotation.id == None,
    #                          and_(GenderUserAnnotation.is_hard == False,
    #                               GenderUserAnnotation.is_bad == False))),
    #                 or_(and_(GenderSample.error > min_error,
    #                          or_(GenderSample.is_annotated_gt,
    #                              GenderUserAnnotation.id != None)),
    #                     and_(GenderSample.error > new_min_error,
    #                          GenderSample.is_annotated_gt==False,
    #                          GenderUserAnnotation.id == None)))).count()

    # new_samples = app.db.session.query(GenderSample). \
    #     outerjoin(GenderUserAnnotation). \
    #     filter(and_(GenderSample.is_checked == False,
    #                 GenderSample.is_bad == False,
    #                 GenderSample.is_hard == False,
    #                 GenderUserAnnotation.id == None,
    #                 or_(GenderSample.send_timestamp == None,
    #                     GenderSample.send_timestamp < expected_utc),
    #                 GenderSample.is_annotated_gt == False,
    #                 GenderSample.error > new_min_error)).count()

    if moderator_permission.can():
        total_reannotated = app.db.session.query(GenderUserAnnotation).count()
    else:
        total_reannotated = 0

    # user_annotated = app.db.session.query(GenderUserLog).filter(GenderUserLog.user_id==current_user.id).count()

    user_annotated = 0
    info = app.db.session.query(GenderUserAnnotationInfo).filter(GenderUserAnnotationInfo.user_id==current_user.id).first()

    if info:
        user_annotated = info.annotated_num

    # slow
    # total_annotated = app.db.session.query(GenderSample).outerjoin(GenderUserAnnotation).\
    #                     filter(or_(GenderUserAnnotation.id != None, GenderSample.is_annotated_gt)).count()

    # slow
    total_annotated = app.db.session.query(GenderSample).\
                        filter(GenderSample.is_annotated_gt).count()

    stats = {}
    stats['total'] = total
    stats['to_check'] = 0 # to_check
    stats['new_samples'] = 0 # new_samples
    stats['total_checked'] = 0 # total_checked
    stats['user_annotated'] = user_annotated

    stats['total_reannotated'] = total_reannotated
    stats['total_annotated'] = total_annotated + total_reannotated # actually wrong value, but need only to check > 0

    return stats

def get_gender_metrics():
    problem_name = 'gender'
    models = app.db.session.query(LearnedModel.id,LearnedModel.finished_ts,
                                  LearnedModel.num_samples,AccuracyMetric.accuracy).\
        filter(and_(LearnedModel.k_fold==None,
                    LearnedModel.problem_name==problem_name,
                    LearnedModel.finished_ts!=None)).\
        outerjoin(AccuracyMetric).order_by(LearnedModel.id).all()

    metrics_data = {}
    metrics_data['metrics_names'] = [gettext('Accuracy')]
    metrics_data['data'] = []
    for m in models:
        item = {}
        item['model_id'] = m.id
        item['finished_ts'] = m.finished_ts.format('YYYY-MM-DD HH:mm:ss')
        item['num_samples'] = m.num_samples
        item['metrics_values'] = [m.accuracy]
        metrics_data['data'].append(item)

    return metrics_data

def get_gender_user_control_data():

    users = app.db.session.query(User).all()

    user_control = {}
    user_control['data'] = []
    for user in users:
        item = {}

        user_info = app.db.session.query(GenderUserAnnotationInfo).\
            filter(GenderUserAnnotationInfo.user_id==user.id).first()
        if not user_info:
            item['user_id'] = user.id
            item['user_email'] = user.email
            item['ann_accuracy'] = 0
            item['ann_count'] = 0
        else:
            item['user_id'] = user.id
            item['user_email'] = user.email
            item['ann_count'] = user_info.annotated_num

            control_num = user_info.control_num
            correct_num = user_info.correct_num

            if control_num < 5:
                ann_accuracy = 0
            else:
                ann_accuracy = 100. * float(correct_num) / control_num

            item['ann_accuracy'] = ann_accuracy

        user_control['data'].append(item)

    return user_control

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

    prob_distr = np.array([0.5, 0.4, 0.1])
    max_checked_count = app.config.get('CHECKED_TIMES_MAX')

    samples_out = []
    is_male_out = None
    target_count = 21
    changed_gender = False
    verify_sample_added = False

    start = time.time()
    while True:

        sum = prob_distr.sum()
        if sum < 1e-9 and len(samples_out) > 0:
            break
        elif sum < 1e-9 and len(samples_out) == 0 and changed_gender:
            break
        elif sum < 1e-9 and len(samples_out) == 0:
            changed_gender = True
            is_male_out = not is_male_out

        thres = np.cumsum(prob_distr)
        value = np.random.uniform(0, sum)

        # samples, is_male = get_err_test_gender_samples(is_male_spec=is_male_out)
        # is_male_out = is_male if is_male_out is None else is_male_out
        # samples_out.extend(samples)
        # break

        if value < thres[0]:
            # annotated images with high error and no check before
            limit = target_count - len(samples_out)
            samples, is_male = get_err_gender_samples(max_checked_count=0, limit=limit, is_male_spec=is_male_out)
            is_male_out = is_male if is_male_out is None else is_male_out

            if len(samples) < limit:
                prob_distr[0] = 0

            # verify sample
            verify_cnt = 0
            if len(samples) > 2 and not verify_sample_added:
                verify_sample = get_verify_gender_sample(is_male, False)
                if verify_sample is not None:
                    verify_cnt = 1
                    samples.insert(random.randint(0, len(samples)-1), verify_sample)
                    samples = samples[:-1]
                    verify_sample_added = True

            print('is_male={} n={}, to verify={}, error samples with no previous check'.\
                  format(int(is_male_out), len(samples), verify_cnt))
            samples_out.extend(samples)

            if len(samples_out) == target_count:
                break

        elif value >= thres[0] and value < thres[1]:

            # new samples
            limit = target_count - len(samples_out)
            samples, is_male = get_new_gender_samples(limit=limit, is_male_spec=is_male_out)
            is_male_out = is_male if is_male_out is None else is_male_out

            if len(samples) < limit:
                prob_distr[1] = 0

            # verify sample
            verify_cnt = 0
            if len(samples) > 2 and not verify_sample_added:
                verify_sample = get_verify_gender_sample(is_male, True)
                if verify_sample is not None:
                    verify_cnt = 1
                    samples.insert(random.randint(0, len(samples) - 1), verify_sample)
                    samples = samples[:-1]
                    verify_sample_added = True

            print('is_male={} n={}, to verify={}, new samples'. \
                  format(int(is_male_out), len(samples), verify_cnt))

            samples_out.extend(samples)

            if len(samples_out) == target_count:
                break

        elif value >= thres[1] and value < thres[2]:
            # annotated images with high error and check before
            limit = target_count - len(samples_out)
            samples, is_male = get_err_gender_samples(max_checked_count=max_checked_count, limit=limit,
                                                      is_male_spec=is_male_out)
            is_male_out = is_male if is_male_out is None else is_male_out

            if len(samples) < limit:
                prob_distr[2] = 0

            # verify sample
            verify_cnt = 0
            if len(samples) > 2 and not verify_sample_added:
                verify_sample = get_verify_gender_sample(is_male, False)
                if verify_sample is not None:
                    verify_cnt = 1
                    samples.insert(random.randint(0, len(samples) - 1), verify_sample)
                    samples = samples[:-1]
                    verify_sample_added = True

            print('is_male={} n={}, to verify={}, error samples with previous check'. \
                  format(int(is_male_out), len(samples), verify_cnt))
            samples_out.extend(samples)

            if len(samples_out) == target_count:
                break

    elapsed = (time.time() - start) * 1000
    print('getting samples for annotation finished in {:.3f}ms'.format(elapsed))

    return samples_out, is_male_out

def get_new_gender_samples(is_male_spec=None, limit=21):
    utc = arrow.utcnow()
    expected_utc = utc.shift(minutes=-app.config.get('SEND_EXPIRE_MIN'))
    is_male_global = random.randint(0, 1) if is_male_spec is None else is_male_spec
    min_error = app.config.get('NEW_SAMPLES_MIN_ERROR')

    ann = []
    for i in range(2):
        ann = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.error >= min_error,
                        GenderSample.is_checked == False,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.always_test == False, # no test samples
                        or_(GenderSample.send_timestamp == None,
                            GenderSample.send_timestamp < expected_utc),
                        GenderSample.is_male == is_male_global)). \
            outerjoin(GenderUserAnnotation). \
            filter(and_(GenderUserAnnotation.id == None,  # no gt and user annotation
                        GenderSample.is_annotated_gt==False)).order_by(desc(GenderSample.error)).limit(limit).all()

        if len(ann) > 0 or is_male_spec is not None:
            break
        is_male_global = not is_male_global

    utc = arrow.utcnow()
    samples_data = []
    for sample, is_male in ann:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['is_verify'] = False
        sample_data['error'] = sample.error
        sample_data['error_label'] = gettext('uncertainty')

        sample.is_send = True
        sample.send_timestamp = utc

        samples_data.append(sample_data)

    app.db.session.flush()
    app.db.session.commit()

    return samples_data, is_male_global

def get_err_gender_samples(max_checked_count=0, min_checked_count=0, is_male_spec=None, limit=21):

    utc = arrow.utcnow()
    expected_utc = utc.shift(minutes=-app.config.get('SEND_EXPIRE_MIN'))

    is_male_global = random.randint(0, 1) if is_male_spec is None else is_male_spec

    min_error = app.config.get('SAMPLES_MIN_ERROR')

    ann = []
    for i in range(2):
        ann = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.error >= min_error,
                        GenderSample.checked_times <= max_checked_count,
                        GenderSample.checked_times >= min_checked_count,
                        or_(GenderSample.send_timestamp == None,
                            GenderSample.send_timestamp < expected_utc),
                        GenderSample.is_checked == False,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False)). \
            outerjoin(GenderUserAnnotation). \
            filter(and_(GenderUserAnnotation.id != None,
                        GenderUserAnnotation.is_male == is_male_global,
                        GenderUserAnnotation.is_hard == False,
                        GenderUserAnnotation.is_bad == False)).order_by(desc(GenderSample.error)).limit(limit).all()

            # filter(
            # or_(and_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
            #          GenderSample.is_male == is_male_global,
            #          GenderSample.is_annotated_gt),
            #     and_(GenderUserAnnotation.id != None,
            #          GenderUserAnnotation.is_male == is_male_global,
            #          GenderUserAnnotation.is_hard == False,
            #          GenderUserAnnotation.is_bad == False))).order_by(desc(GenderSample.error)).limit(limit).all()

        if len(ann) > 0 or is_male_spec is not None:
            break
        is_male_global = not is_male_global

    utc = arrow.utcnow()
    samples_data = []
    for sample, is_male in ann:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['is_verify'] = False
        sample_data['error'] = sample.error
        sample_data['error_label'] = gettext('error')

        sample.is_send = True
        sample.send_timestamp = utc

        samples_data.append(sample_data)

    app.db.session.flush()
    app.db.session.commit()

    return samples_data, is_male_global

def get_err_test_gender_samples(is_male_spec=None):

    utc = arrow.utcnow()
    expected_utc = utc.shift(minutes=-app.config.get('SEND_EXPIRE_MIN'))

    is_male_global = random.randint(0, 1) if is_male_spec is None else is_male_spec

    min_error = app.config.get('SAMPLES_MIN_ERROR')

    ann = []
    for i in range(2):
        ann = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.error >= 0.5,
                        GenderSample.always_test == True,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False)). \
            outerjoin(GenderUserAnnotation). \
            filter(
            or_(and_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
                     GenderSample.is_male == is_male_global,
                     GenderSample.is_annotated_gt),
                and_(GenderUserAnnotation.id != None,
                     GenderUserAnnotation.is_male == is_male_global,
                     GenderUserAnnotation.is_hard == False,
                     GenderUserAnnotation.is_bad == False))).order_by(desc(GenderSample.error)).all()

        if len(ann) > 0 or is_male_spec is not None:
            break
        is_male_global = not is_male_global

    utc = arrow.utcnow()
    samples_data = []
    for sample, is_male in ann:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['is_verify'] = False
        sample_data['error'] = sample.error
        sample_data['error_label'] = gettext('error')

        sample.is_send = True
        sample.send_timestamp = utc

        samples_data.append(sample_data)

    app.db.session.flush()
    app.db.session.commit()

    return samples_data, is_male_global

def get_verify_gender_sample(is_male, is_new):

    is_male = random.randint(0, 1)

    max_error_samples = app.config.get('VERIFY_SAMPLES_MAX_ERROR')
    max_error_new_samples = app.config.get('VERIFY_NEW_SAMPLES_MAX_ERROR')

    if not is_new:
        sample = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.error < max_error_samples,
                        GenderSample.is_male == is_male,
                        GenderSample.is_annotated_gt,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False)).order_by(func.rand()).limit(1).first()
    else:
        sample = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.error < max_error_new_samples,
                        GenderSample.is_male == is_male,
                        GenderSample.is_annotated_gt == False,
                        GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False)).order_by(func.rand()).limit(1).first()

    sample_data = None
    if sample is not None:
        sample_data = {}
        sample_data['is_male'] = is_male
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['is_verify'] = True
        sample_data['error'] = sample.error
        sample_data['error_label'] = gettext('uncertainty') + "*" if is_new else gettext('error') + "*"

    return sample_data

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
        args['is_verify'] = False
        if 'is_verify' in data_item and check_is_int(data_item['is_verify']):
            args['is_verify'] = data_item['is_verify'] != 0
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

def get_gender_test_errors():
    limit = 1000
    res = app.db.session.query(GenderSample,
                               case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                    else_=GenderUserAnnotation.is_male),
                               case([(GenderUserAnnotation.id == None, GenderSample.is_hard)],
                                    else_=GenderUserAnnotation.is_hard),
                               case([(GenderUserAnnotation.id == None, GenderSample.is_bad)],
                                    else_=GenderUserAnnotation.is_bad)). \
        filter(and_(GenderSample.error > 0.5,
                    GenderSample.always_test == True)). \
        outerjoin(GenderUserAnnotation). \
        filter(or_(GenderUserAnnotation.id != None,  # annotated or gt
                   GenderSample.is_annotated_gt == True)).order_by(desc(GenderSample.error)).limit(limit).all()

    samples_data = []
    for sample, is_male, is_hard, is_bad in res:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['is_bad'] = int(is_bad or is_hard)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['error'] = sample.error
        sample_data['error_label'] = gettext('error')
        if is_hard or is_bad:
            continue
        samples_data.append(sample_data)

    return samples_data

def get_unsure_samples():
    limit = 1000
    res = app.db.session.query(GenderSample,
                               GenderSampleResult.prob_pos,
                               case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                    else_=GenderUserAnnotation.is_male),
                               case([(GenderUserAnnotation.id == None, GenderSample.is_hard)],
                                    else_=GenderUserAnnotation.is_hard),
                               case([(GenderUserAnnotation.id == None, GenderSample.is_bad)],
                                    else_=GenderUserAnnotation.is_bad)). \
        outerjoin(GenderUserAnnotation). \
        outerjoin(GenderSampleResult). \
        filter(and_(GenderSample.always_test == False,
                    GenderUserAnnotation.id != None,
                    GenderSampleResult.prob_pos > 0.49,
                    GenderSampleResult.prob_pos < 0.51)).limit(limit).all()

    samples_data = []

    for sample, prob_male, is_male, is_hard, is_bad in res:
        sample_data = {}
        sample_data['is_male'] = int(is_male)
        sample_data['is_bad'] = int(is_bad or is_hard)
        sample_data['image'] = url_for('image', id=sample.image.id)
        sample_data['id'] = sample.id
        sample_data['error'] = 1 - prob_male if prob_male > 0.5 else prob_male
        sample_data['error_label'] = gettext('uncertainty')
        if is_hard or is_bad:
            continue
        samples_data.append(sample_data)

    return samples_data

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
    moderator_permission.test(403)
    if problem not in ['gender']:
        abort(404)
    ctx = {'ts': int(time.time())}
    if problem == 'gender':
        metrics = get_gender_metrics()
        return render_template('metrics.html', metrics=metrics, ctx=ctx)
    else:
        abort(404)

@app.route('/control/<string:problem>', methods=['GET'])
@login_required
@nocache
def user_control(problem):
    moderator_permission.test(403)
    if problem not in ['gender']:
        abort(404)
    ctx = {'ts': int(time.time())}

    if problem == 'gender':
        user_control = get_gender_user_control_data()
        return render_template('user_control.html', user_control=user_control, ctx=ctx)
    else:
        abort(404)

@app.route('/test_errors/<string:problem>', methods=['GET'])
@login_required
@nocache
def test_errors(problem):
    moderator_permission.test(403)
    if problem not in ['gender']:
        abort(404)
    ctx = {'ts': int(time.time())}
    if problem == 'gender':
        ctx['samples'] = get_gender_test_errors()
        ctx['is_empty'] = len(ctx['samples']) == 0
        return render_template('samples_view.html', ctx=ctx)
    else:
        abort(404)

@app.route('/unsure_samples/<string:problem>', methods=['GET'])
@login_required
@nocache
def unsure_samples(problem):
    moderator_permission.test(403)
    if problem not in ['gender']:
        abort(404)
    ctx = {'ts': int(time.time())}
    if problem == 'gender':
        ctx['samples'] = get_unsure_samples()
        ctx['is_empty'] = len(ctx['samples']) == 0
        return render_template('samples_view.html', ctx=ctx)
    else:
        abort(404)

@app.route('/update_gender_data', methods=['POST'])
@login_required
@nocache
def update_gender_data():
    form = GenderDataForm(request.form)
    failed = True

    verify_new_samples_max_err = app.config.get('VERIFY_NEW_SAMPLES_MAX_ERROR')
    verify_samples_max_err = app.config.get('VERIFY_SAMPLES_MAX_ERROR')

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
            print('input data is invalid')
            break
        # filt_gender_data = gender_filter_only_changed(gender_data)

        list_of_ids = [sample_id for sample_id in gender_data]

        # select samples data
        samples_data = app.db.session.query(GenderSample, GenderUserAnnotation).filter(GenderSample.id.in_(list_of_ids))\
            .outerjoin(GenderUserAnnotation).all()
        samples_db_data = {sd.GenderSample.id:sd for sd in samples_data}

        if len(samples_db_data) != len(list_of_ids):
            print('wrong samples ids')
            break

        n_changed = 0
        n_not_changed = 0
        control_num = 0
        correct_num = 0
        annotated_num = 0

        for sample_id in list_of_ids:
            gdata = gender_data[sample_id]
            sample_db = samples_db_data[sample_id].GenderSample
            user_ann_db = samples_db_data[sample_id].GenderUserAnnotation

            is_verify = gdata['is_verify']
            is_changed = False

            if user_ann_db is None:

                if is_verify:
                    # verification of user annotation precision
                    control_num += 1

                    db_is_hard = sample_db.is_hard or sample_db.is_bad
                    ann_is_hard = gdata['is_hard'] or gdata['is_bad']

                    is_wrong = False
                    if db_is_hard != ann_is_hard:
                        is_wrong = True
                    if not db_is_hard and gdata['is_male'] != sample_db.is_male:
                        is_wrong = True

                    correct_num += 1 if not is_wrong else 0
                else:
                    # user annotation
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
                if is_verify:

                    control_num += 1

                    db_is_hard = user_ann_db.is_hard or user_ann_db.is_bad
                    ann_is_hard = gdata['is_hard'] or gdata['is_bad']

                    is_wrong = False
                    if db_is_hard != ann_is_hard:
                        is_wrong = True
                    if not db_is_hard and gdata['is_male'] != user_ann_db.is_male:
                        is_wrong = True

                    correct_num += 1 if not is_wrong else 0
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

            if not is_verify:
                if is_changed:
                    n_changed += 1
                    sample_db.checked_times = 0
                    sample_db.changed_ts = utc
                    sample_db.is_changed = True
                else:
                    n_not_changed += 1
                    sample_db.checked_times += 1

            user_log = GenderUserLog(sample_id=sample_db.id, user_id=user_id, mark_timestamp=utc)
            app.db.session.add(user_log)
            app.db.session.flush()

            sample_db.is_checked = True
            sample_db.is_send = False

            annotated_num += 1

        user_info = app.db.session.query(GenderUserAnnotationInfo).filter(GenderUserAnnotationInfo.user_id==user_id).first()
        if not user_info:
            user_info = GenderUserAnnotationInfo(user_id=user_id)
            app.db.session.add(user_info)
            app.db.session.flush()

        user_info.control_num += control_num
        user_info.correct_num += correct_num
        user_info.annotated_num += annotated_num

        print('changed: {}, not changed: {}, control: {}, correct: {}'.format(n_changed,
                                                                              n_not_changed,
                                                                              control_num, correct_num))

        app.db.session.flush()
        app.db.session.commit()
        failed = False
        break

    if failed:
        print('failed to update')
        flash('Failed to update data.')
    else:
        print('successfully updated data')
        # flash('Data successfully updated: added: {}, updated: {}, deleted: {}'.format(added, updated, deleted))
    return redirect(url_for('reannotation', problem='gender'))

@app.route('/trigger_train/<problem_name>')
@login_required
@nocache
def trigger_train(problem_name):
    admin_permission.test(403)
    if problem_name == 'gender':
        task_id = trigger_train_gender()
        if task_id is None:
            return jsonify(status='error', message='failed to start training'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), task_id=task_id, message='training started.'), 202

def trigger_train_gender():
    problem_name = 'gender'
    if check_working_tasks(problem_name, 'train'):
        print('attempted to start training while other task not finished')
        return None

    clear_old_tasks(problem_name, 'train')

    task_id = celery.uuid()
    utc = arrow.utcnow()
    task_db = LearningTask(problem_name=problem_name, problem_type='train',
                           task_id=task_id, started_ts=utc)
    app.db.session.add(task_db)
    app.db.session.flush()
    app.db.session.commit()

    task = run_train.apply_async((problem_name,), task_id=task_id,
                                 link_error=train_on_error.s(), link=train_on_success.s(),
                                 queue='learning')

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_train/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_train(task_id):
    admin_permission.test(403)
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
    admin_permission.test(403)
    if problem_name == 'gender':
        task_id = trigger_test_gender()
        if task_id is None:
            return jsonify(status='error', message='failed to start testing'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), task_id=task_id, message='testing started.'), 202

def trigger_test_gender():
    problem_name = 'gender'
    if check_working_tasks(problem_name, 'test'):
        print('attempted to start testing while other task not finished')
        return None

    if get_learned_models_count(problem_name) == 0:
        print('no models for testing')
        return None

    clear_old_tasks(problem_name, 'test')

    task_id = celery.uuid()
    utc = arrow.utcnow()
    task_db = LearningTask(problem_name=problem_name,problem_type='test',
                           task_id=task_id,started_ts=utc)
    app.db.session.add(task_db)
    app.db.session.flush()
    app.db.session.commit()

    task = run_test.apply_async((problem_name,), task_id=task_id,
                                link_error=test_on_error.s(), link=test_on_success.s(),
                                queue='learning')

    print('{} task successfully started'.format(task.id))
    return task.id

@app.route('/stop_test/<string:task_id>', methods=['GET', 'POST'])
@login_required
def stop_test(task_id):
    admin_permission.test(403)
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
    admin_permission.test(403)
    if problem_name == 'gender':
        task_ids = trigger_train_k_folds_gender()
        if task_ids is None:
            return jsonify(status='error', message='failed to start training k-folds'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), tasks_id=task_ids, message='training started.'), 202

def trigger_train_k_folds_gender():
    problem_name = 'gender'
    problem_type = 'train_k_folds'

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    for k_fold in range(k_folds):

        if check_working_tasks(problem_name, problem_type, k_fold=k_fold):
            print('k-fold {}: attempted to start training while other task not finished'.format(k_fold))
            continue

        clear_old_tasks(problem_name, problem_type, k_fold=k_fold)

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name,problem_type=problem_type,
                               task_id=task_id,started_ts=utc,k_fold=k_fold)

        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_train_k_folds.apply_async((problem_name, k_fold), task_id=task_id,
                                             link_error=train_k_folds_on_error.s(),
                                             link=train_k_folds_on_success.s(),
                                             queue='learning')
        task_ids.append(task_id)

    if len(task_ids) > 0:
        print('{} tasks successfully started'.format(len(task_ids)))
    return task_ids

@app.route('/stop_train_k_folds/<string:task_ids_str>', methods=['GET', 'POST'])
@login_required
def stop_train_k_folds(task_ids_str):
    admin_permission.test(403)
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
    admin_permission.test(403)
    if problem_name == 'gender':
        task_ids = trigger_test_k_folds_gender()
        if task_ids is None:
            return jsonify(status='error', message='failed to start training k-folds'), 400
    else:
        return jsonify(status='error', message='wrong post parameters'), 400

    return jsonify(status='ok', problems=get_problems(), tasks_id=task_ids, message='training started.'), 202

def trigger_test_k_folds_gender():
    problem_name = 'gender'
    problem_type = 'test_k_folds'

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    for k_fold in range(k_folds):

        if check_working_tasks(problem_name, problem_type, k_fold=k_fold):
            print('k-fold {}: attempted to start training while other task not finished'.format(k_fold))
            continue

        if get_learned_models_count(problem_name, k_fold=k_fold) == 0:
            print('k-fold {}: no models for testing'.format(k_fold))
            continue

        clear_old_tasks(problem_name, problem_type, k_fold=k_fold)

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name,problem_type=problem_type,
                               task_id=task_id,started_ts=utc,k_fold=k_fold)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_test_k_folds.apply_async((problem_name, k_fold), task_id=task_id,
                                            link_error=test_k_folds_on_error.s(),
                                            link=test_k_folds_on_success.s(),
                                            queue='learning')
        task_ids.append(task_id)

    if len(task_ids) > 0:
        print('{} tasks successfully started'.format(len(task_ids)))
    return task_ids

@app.route('/stop_test_k_folds/<string:task_ids_str>', methods=['GET', 'POST'])
@login_required
def stop_test_k_folds(task_ids_str):
    admin_permission.test(403)
    task_ids = task_ids_str.split(',')
    for task_id in task_ids:
        task = LearningTask.query.filter_by(task_id=task_id).first()
        if task and task.finished_ts != None:
            continue
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
    admin_permission.test(403)
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

@app.route('/reannotation_data/')
@login_required
def reannotation_data():
    response = get_problems()
    return jsonify(response)
