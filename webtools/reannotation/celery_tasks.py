# -*- coding: utf-8 -*-

from webtools import app
import time
import random
from os.path import join, isdir
from os import mkdir, makedirs
from sqlalchemy import func, or_, and_, desc, not_
from models import GenderSample, LearningTask, LearnedModel, AccuracyMetric, GenderUserAnnotation, GenderSampleResult
from shutil import copyfile
from ..utils import add_path
from sqlalchemy.sql.expression import case
import sys
import arrow

import numpy as np

class StatusUpdater:
    def __init__(self, task_id):
        self.task = LearningTask.query.filter_by(task_id=task_id)
        self.rescales = []
        self.prefix = []

    def update_state(self, state='PENDING', progress=0.0, status='Pending..'):
        task = self.task.first()
        if task:
            task.state = state
            task.status = self.__prefix_status(status)
            task.progress = self.__rescaled_progress(progress)
            app.db.session.flush()
            app.db.session.commit()

    def finish(self, ts, progress=1.0):
        task = self.task.first()
        if task:
            task.finished_ts = ts
            task.progress = self.__rescaled_progress(progress)
            app.db.session.flush()
            app.db.session.commit()

    def __rescaled_progress(self, progress):
        progress_n = progress
        for shift, scale in self.rescales[::-1]:
            progress_n = progress_n * scale + shift
        return progress_n

    def __prefix_status(self, status):
        status_n = status
        for prefix in self.prefix[::-1]:
            status_n = prefix + status_n
        return status_n

    def push_rescale(self, shift, scale):
        self.rescales.append((shift, scale))

    def pop_rescale(self):
        self.rescales.pop()

    def push_prefix(self, prefix):
        self.prefix.append(prefix)

    def pop_prefix(self):
        self.prefix.pop()

def dump_result():
    return {'current': 100, 'total': 100, 'status': 'Task completed',
            'result': 0}

@app.celery.task(bind=True)
def dump_task(self):
    return {'current': 100, 'total': 100, 'status': 'Task completed.',
            'result': 0}

def get_test_samples(for_model_id, k_fold=None):

    subq = app.db.session.query(GenderSampleResult).\
        filter(GenderSampleResult.model_id == for_model_id).subquery('t')

    if k_fold is not None:
        # (sample_id, is_male (picked from annotation or gt), prob female, prob male)
        res = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.k_fold != None,
                        GenderSample.k_fold == k_fold)). \
            outerjoin(GenderUserAnnotation). \
            filter(or_(GenderUserAnnotation.id == None,
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))). \
            outerjoin(subq). \
            filter(subq.c.id == None).all()
    else:
        # (sample_id, is_male (picked from annotation or gt), prob female, prob male)
        res = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.always_test == True)). \
            outerjoin(GenderUserAnnotation). \
            filter(or_(and_(GenderUserAnnotation.id == None,
                            GenderSample.is_annotated_gt),
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))).\
            outerjoin(subq).\
            filter(subq.c.id==None).all()

    samples = [(s.GenderSample.image.filename(), 1 if s[1] else 0, s.GenderSample.id) for s in res]

    # app.db.session.query(GenderSampleResult, GenderSample, GenderUserAnnotation).\
    #     filter(GenderSampleResult.model_id==for_model_id).\
    #     outerjoin(GenderSample). \
    #     filter(and_(GenderSample.is_hard == False,
    #                 GenderSample.is_bad == False,
    #                 GenderSample.is_annotated_gt,
    #                 GenderSample.always_test == True)). \
    #     outerjoin(GenderUserAnnotation). \
    #     filter(GenderUserAnnotation.id == None). \
    #     order_by(func.random()).all()
    #
    # samples_gt = app.db.session.query(GenderSample). \
    #     filter(and_(GenderSample.is_hard == False,
    #                 GenderSample.is_bad == False,
    #                 GenderSample.is_annotated_gt,
    #                 GenderSample.always_test == True)). \
    #     outerjoin(GenderUserAnnotation). \
    #     filter(GenderUserAnnotation.id == None). \
    #     outerjoin(GenderSampleResult). \
    #     filter(or_(GenderSampleResult.model_id != for_model_id,
    #                GenderSampleResult.id == None)). \
    #     order_by(func.random()).all()
    #
    # samples_ann = app.db.session.query(GenderSample, GenderUserAnnotation). \
    #     filter(GenderSample.always_test == True). \
    #     join(GenderUserAnnotation). \
    #     filter(and_(GenderUserAnnotation.is_hard == False,
    #                 GenderUserAnnotation.is_bad == False)). \
    #     outerjoin(GenderSampleResult). \
    #     filter(or_(GenderSampleResult.model_id != for_model_id,
    #                 GenderSampleResult.id == None)). \
    #     order_by(func.random()).all()

    # samples = [(s.image.filename(), 1 if s.is_male else 0, s.id) for s in samples_gt]
    # samples.extend([(s.image.filename(), 1 if s.GenderUserAnnotation.is_male else 0, s.id) for s in samples_ann])

    return samples

def get_train_samples(k_fold=None):

    if k_fold is not None:
        # (sample, is_male (picked from annotation or gt))
        res = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.always_test == False,
                        GenderSample.k_fold != None,
                        GenderSample.k_fold != k_fold)). \
            outerjoin(GenderUserAnnotation). \
            filter(or_(and_(GenderUserAnnotation.id == None,
                            GenderSample.is_annotated_gt),
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))).all()
    else:
        # (sample, is_male (picked from annotation or gt))
        res = app.db.session.query(GenderSample,
                                   case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                        else_=GenderUserAnnotation.is_male)). \
            filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                        GenderSample.is_bad == False,
                        GenderSample.always_test == False)). \
            outerjoin(GenderUserAnnotation). \
            filter(or_(and_(GenderUserAnnotation.id == None,
                            GenderSample.is_annotated_gt),
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))).all()

    samples = [(s.GenderSample.image.filename(), 1 if s[1] else 0, s.GenderSample.id) for s in res]

    return samples

def update_gender_cv_partition():

    n_samples = GenderSample.query \
        .filter_by(always_test=False,k_fold=None).count()

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
            .order_by(func.random())\
            .limit(cnt).with_entities(GenderSample.id)
        subset_to_update = GenderSample.query.filter(GenderSample.id.in_(subset))
        subset_to_update.update(dict(k_fold=k_fold), synchronize_session='fetch')
        app.db.session.flush()

    app.db.session.commit()

# train
@app.celery.task(bind=True)
def run_train(self, taskname):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        run_gender_train(updater, self.request.id)
        updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')
        updater.finish(arrow.now())
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_train(updater, task_id, k_fold=None):

    updater.update_state(state='PROGRESS', progress=0.005, status='Preparing samples for training..')

    samples = get_train_samples(k_fold=k_fold)
    print('number of samples: {}'.format(len(samples)))

    if len(samples) == 0:
        updater.update_state(state='FAILURE', progress=1.0, status='No samples to train on')
        updater.finish(arrow.now())
        return None

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing training scripts..')
    updater.push_rescale(0.01, 0.99)

    # create model
    gender_model = LearnedModel(task_id=task_id, problem_name='gender')
    app.db.session.add(gender_model)
    app.db.session.flush()
    exp_num = gender_model.id

    trainroom_dir = app.config.get('TRAINROOM_FOLDER')
    trainroom_gender = join(trainroom_dir, 'gender')

    # prepare experiment directory
    exp_fold = 'fold{}'.format(k_fold) if k_fold is not None else 'main'
    exp_dir = join(trainroom_gender, 'exps', 'exp{}_{}'.format(exp_num, exp_fold))
    exp_base = join(trainroom_gender, 'exps', 'base_exp')

    # copy scripts for training
    if not isdir(exp_dir):
        makedirs(exp_dir)
    copyfile(join(exp_base, 'config.yml'), join(exp_dir, 'config.yml'))
    copyfile(join(exp_base, 'get_model.py'), join(exp_dir, 'get_model.py'))
    copyfile(join(exp_base, 'get_optimizer_params.py'), join(exp_dir, 'get_optimizer_params.py'))

    # run training
    with add_path(trainroom_gender):
        solve_module = __import__('solve')
        snapshot_prefix, epoch = solve_module.solve(updater, samples, [], trainroom_dir, exp_dir)
        del sys.modules['solve']

    # udpate model in db
    model = LearnedModel.query.filter_by(task_id=task_id,problem_name='gender').first()
    model.exp_dir = exp_dir
    model.prefix = snapshot_prefix
    model.epoch = epoch
    if k_fold is not None:
        model.k_fold = k_fold
    model.finished_ts = arrow.now()
    app.db.session.flush()
    app.db.session.commit()

    updater.pop_rescale()

    return model.id

@app.celery.task
def train_on_error(uuid):
    clear_data_for_train_task(uuid, 'FAILURE', 'Error occurred while training')
    return {'current': 1, 'total': 1, 'status': 'Error during training occured',
                'result': 0}

@app.celery.task
def train_on_success(result):
    print('run_train successfully finished')
    return {'current': 1, 'total': 1, 'status': 'Training successfully finished',
            'result': 0}

def clear_data_for_train_task(uuid, state, status):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    gender_model = LearnedModel.query.filter_by(task_id=uuid)
    gender_model.delete()
    app.db.session.flush()
    app.db.session.commit()

# test
def compute_gender_metrics(gender_model, update_state=True):

    # (sample_id, is_male (picked from annotation or gt), prob female, prob male)
    res = app.db.session.query(GenderSample.id,
                               case([(GenderUserAnnotation.id==None, GenderSample.is_male)],
                                    else_=GenderUserAnnotation.is_male),
                               GenderSampleResult.prob_neg,
                               GenderSampleResult.prob_pos).\
        filter(and_(GenderSample.is_hard == False, # no bad or hard samples
                    GenderSample.is_bad == False)).\
        outerjoin(GenderUserAnnotation). \
        filter(or_(and_(GenderUserAnnotation.id==None, # if sample has annotation, check it is not marked as hard or bad
                        GenderSample.is_annotated_gt),
                   and_(GenderUserAnnotation.id!=None,
                        GenderUserAnnotation.is_hard == False,
                        GenderUserAnnotation.is_bad == False))).\
        outerjoin(GenderSampleResult).\
        filter(and_(GenderSampleResult.id!=None,
                    GenderSampleResult.model_id==gender_model.id)).all()

    n_correct = 0
    n_samples = 0
    for id, is_male, prob_neg, prob_pos in res:

        if prob_pos > prob_neg:
            n_correct += 1 if is_male else 0
        else:
            n_correct += 1 if not is_male else 0
        n_samples += 1

    metric_value = 0.0
    if n_samples > 0:
        metric_value = float(n_correct) / n_samples

    metric_name = 'accuracy'

    if update_state:
        metric = AccuracyMetric.query.filter_by(model_id=gender_model.id)
        if metric.count() > 0:
            metric.update(dict(accuracy=metric_value))
        else:
            metric = AccuracyMetric(model_id=gender_model.id,
                                    accuracy=metric_value)
            app.db.session.add(metric)

        app.db.session.flush()
        app.db.session.commit()

    return metric_name, metric_value

def compute_errors(gender_model):

    # (sample_id, is_male (picked from annotation or gt), prob female, prob male)
    ann = app.db.session.query(GenderSample,
                               case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                    else_=GenderUserAnnotation.is_male),
                               GenderSample.checked_times,
                               GenderSampleResult.prob_neg,
                               GenderSampleResult.prob_pos). \
        filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                    GenderSample.is_bad == False)). \
        outerjoin(GenderUserAnnotation). \
        filter(or_(GenderUserAnnotation.id == None,  # if sample has annotation check it is not marked as hard or bad
                   and_(GenderUserAnnotation.is_hard == False,
                        GenderUserAnnotation.is_bad == False))). \
        outerjoin(GenderSampleResult). \
        filter(and_(GenderSampleResult.id != None,
                    GenderSampleResult.model_id == gender_model.id)).all()

    checked_times_max = app.config.get('CHECKED_TIMES_MAX')
    checked_times_coeff = app.config.get('CHECKED_TIMES_COEFF')
    checked_times_min = app.config.get('CHECKED_TIMES_MIN')

    for sample, is_male, checked_times, prob_neg, prob_pos in ann:
        err = prob_neg if is_male else prob_pos
        sample.error = err
        sample.is_checked = False

    app.db.session.flush()
    app.db.session.commit()

@app.celery.task(bind=True)
def run_test(self, taskname):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)

        gender_models = LearnedModel.query.\
            filter(and_(LearnedModel.problem_name=='gender',
                        LearnedModel.k_fold==None,
                        LearnedModel.prefix!=None)). \
            order_by(desc(LearnedModel.id)).all()

        num_models = len(gender_models)

        if num_models == 0:
            updater.update_state(state='FAILURE', progress=1.0, status='No models to test')
            updater.finish(arrow.now())
            return dump_result()

        last_metric_name = ''
        last_metric_value = 0.0

        for idx, gender_model in enumerate(gender_models):
            scale = 1.0 / num_models
            shift = idx * scale
            updater.push_rescale(shift, scale)
            updater.push_prefix('Model [{}/{}] '.format(idx + 1, num_models))

            run_gender_test(updater, gender_model)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing metrics..')
            metric_name, metric_value = compute_gender_metrics(gender_model)

            if idx == 0:
                updater.update_state(state='PROGRESS', progress=1.0,
                                     status='Computing errors..')
                compute_errors(gender_model)

                last_metric_name = metric_name
                last_metric_value = metric_value

            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='Successfully tested, {}={:.3f}'.format(metric_name, metric_value))

            updater.pop_prefix()
            updater.pop_rescale()

        updater.update_state(state='SUCCESS', progress=1.0,
                             status='Successfully finished, {}={:.3f}'.format(last_metric_name, last_metric_value))

        updater.finish(arrow.now())
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_test(updater, gender_model):

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing samples for testing..')
    samples = get_test_samples(gender_model.id, k_fold=gender_model.k_fold)
    print('number of samples: {}'.format(len(samples)))

    if len(samples) == 0:
        updater.update_state(state='PROGRESS', progress=1.0, status='No samples to test')
        return None

    # run training
    trainroom_dir = app.config.get('TRAINROOM_FOLDER')
    trainroom_gender = join(trainroom_dir, 'gender')

    snapshot = str(gender_model.prefix)
    epoch = gender_model.epoch
    exp_dir = str(gender_model.exp_dir)

    updater.push_rescale(0.01, 0.99)
    with add_path(trainroom_gender):
        test_module = __import__('test')
        metric_data, pr_probs = test_module.test(updater, snapshot, epoch, samples, exp_dir)
        del sys.modules['test']

    updater.pop_rescale()

    updater.update_state(state='PROGRESS', progress=1.0,
                         status='Storing results..')

    assert(len(samples) == pr_probs.shape[0])
    for i in range(len(samples)):
        pr_prob = pr_probs[i,:]
        prob_neg = pr_prob[0]
        prob_pos = pr_prob[1]
        sample_id = samples[i][2]
        result_sample = GenderSampleResult(sample_id=sample_id,
                                           model_id=gender_model.id, prob_neg=prob_neg, prob_pos=prob_pos)
        app.db.session.add(result_sample)
        app.db.session.flush()

@app.celery.task
def test_on_error(uuid):
    clear_data_for_test_task(uuid, 'FAILURE', 'Error occurred while testing')
    return {'current': 1, 'total': 1, 'status': 'Error during testing occured',
            'result': 0}

@app.celery.task
def test_on_success(result):
    print('run_train successfully finished')
    return {'current': 1, 'total': 1, 'status': 'Testing successfully finished',
            'result': 0}

def clear_data_for_test_task(uuid, state, status):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

# train k-folds
@app.celery.task(bind=True)
def run_train_k_folds(self, taskname, k_fold):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        update_gender_cv_partition()
        run_gender_train(updater, self.request.id, k_fold=k_fold)
        updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')
        updater.finish(arrow.now())
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

@app.celery.task
def train_k_folds_on_error(uuid):
    clear_data_for_train_k_folds_task(uuid, 'FAILURE', 'Error occurred while training')
    return {'current': 1, 'total': 1, 'status': 'Error during training occured.',
            'result': 0}

@app.celery.task
def train_k_folds_on_success(result):
    print('Training successfully finished')
    return {'current': 1, 'total': 1, 'status': 'Training successfully finished',
            'result': 0}

def clear_data_for_train_k_folds_task(uuid, state, status):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    gender_model = LearnedModel.query.filter_by(task_id=uuid)
    gender_model.delete()
    app.db.session.flush()
    app.db.session.commit()

# test k-folds
@app.celery.task(bind=True)
def run_test_k_folds(self, taskname, k_fold):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        update_gender_cv_partition()

        gender_model = LearnedModel.query.filter_by(k_fold=k_fold, problem_name='gender').\
            order_by(desc(LearnedModel.id)).first()

        if len(gender_model) > 0:

            run_gender_test(updater, gender_model)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing metrics..')
            metric_name, metric_value = compute_gender_metrics(gender_model, update_state=False)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing errors..')
            compute_errors(gender_model)

            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='Successfully tested, {}={:.3f}'.format(metric_name, metric_value))
        else:
            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='No model to test')
        updater.finish(arrow.now())

        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

@app.celery.task
def test_k_folds_on_error(uuid):
    clear_data_for_test_k_folds_task(uuid, 'FAILURE', 'Error occurred while testing')
    return {'current': 1, 'total': 1, 'status': 'Error during testing occured.',
            'result': 0}

@app.celery.task
def test_k_folds_on_success(result):
    print('Testing successfully finished')
    return {'current': 1, 'total': 1, 'status': 'Testing successfully finished',
            'result': 0}

def clear_data_for_test_k_folds_task(uuid, state, status):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()