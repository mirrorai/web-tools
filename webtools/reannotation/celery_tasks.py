# -*- coding: utf-8 -*-

from webtools import app
import time
import random
from os.path import join, isdir
from os import mkdir, makedirs
from sqlalchemy import func, or_, and_, desc, not_
from models import GenderSample, LearningTask, LearnedModel, AccuracyMetric, GenderUserAnnotation, GenderSampleResult
from models import GPUStatus
from shutil import copyfile
from ..utils import add_path
from sqlalchemy.sql.expression import case
import sys
import arrow
from threading import Lock
from webtools.utils import send_slack_message
from flask import url_for

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
            filter(or_(GenderUserAnnotation.id == None,
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))).\
            outerjoin(subq).\
            filter(subq.c.id==None).all()

    samples = [(s.GenderSample.image.filename(), 1 if s[1] else 0, s.GenderSample.id) for s in res]

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

    with app.db_lock:
        n_samples = GenderSample.query \
            .filter_by(always_test=False,k_fold=None).count()
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
                .order_by(func.random())\
                .limit(cnt).all()
            for s in subset:
                s.k_fold = k_fold
            # subset_to_update = GenderSample.query.filter(GenderSample.id.in_(subset))
            # subset_to_update = GenderSample.query.join(subset, GenderSample.id==subset.c.id)
            # subset_to_update.update(dict(k_fold=k_fold), synchronize_session='fetch')
            app.db.session.flush()

        app.db.session.commit()

# train
@app.celery.task(bind=True)
def run_train(self, taskname):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        run_gender_train(updater, self.request.id)
        updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')
        updater.finish(arrow.utcnow())
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_train(updater, task_id, k_fold=None):

    updater.update_state(state='PROGRESS', progress=0.01, status='Waiting available gpu..')
    gpu_id = wait_available_gpu(task_id)

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing samples for training..')

    samples = get_train_samples(k_fold=k_fold)
    print('number of samples: {}'.format(len(samples)))

    if len(samples) == 0:
        updater.update_state(state='FAILURE', progress=1.0, status='No samples to train on')
        model_id = None
    else:
        updater.update_state(state='PROGRESS', progress=0.01, status='Preparing training scripts..')
        updater.push_rescale(0.01, 0.99)

        # create model
        gender_model = LearnedModel(task_id=task_id, num_samples=len(samples), k_fold=k_fold,
                                    problem_name='gender', started_ts=arrow.utcnow())
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
            snapshot_prefix, epoch = solve_module.solve(updater, samples, [], trainroom_dir, exp_dir, gpu_id=gpu_id)
            del sys.modules['solve']

        # udpate model in db
        model = LearnedModel.query.filter_by(task_id=task_id,problem_name='gender').first()
        model.exp_dir = exp_dir
        model.prefix = snapshot_prefix
        model.epoch = epoch
        model.finished_ts = arrow.utcnow()
        app.db.session.flush()
        app.db.session.commit()

        updater.pop_rescale()

        model_id = model.id

        # report slack
        report_gender_train(model)

    updater.update_state(state='PROGRESS', progress=1.0, status='Releasing GPU..')
    release_gpu(task_id)

    updater.finish(arrow.now())

    return model_id

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
    ts = arrow.utcnow()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    gender_model = LearnedModel.query.filter_by(task_id=uuid)
    gender_model.delete()
    app.db.session.flush()
    app.db.session.commit()

    release_gpu(uuid)

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
                                    accuracy=metric_value, finished_ts=arrow.utcnow())
            app.db.session.add(metric)

        app.db.session.flush()
        app.db.session.commit()

    return metric_name, metric_value

def compute_errors(gender_model):

    # (sample_id, is_male (picked from annotation or gt), prob female, prob male)
    ann = app.db.session.query(GenderSample,
                               case([(GenderUserAnnotation.id == None, GenderSample.is_male)],
                                    else_=GenderUserAnnotation.is_male),
                               case([(and_(GenderUserAnnotation.id == None,
                                           GenderSample.is_annotated_gt==False), False)], # has annotation
                                    else_=True),
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

    for sample, is_male, is_ann, checked_times, prob_neg, prob_pos in ann:
        if is_ann:
            err = prob_neg if is_male else prob_pos
        else:
            err = min(prob_neg, prob_pos)
            sample.is_male = prob_pos >= prob_neg
        sample.error = err
        sample.is_checked = False
        sample.is_changed = False

    app.db.session.flush()
    app.db.session.commit()

def remove_old_main_models(problem_name):

    keep_top_cnt = app.config.get('KEEP_TOP_MODELS_CNT')

    last_two = app.db.session.query(LearnedModel).\
        filter(and_(LearnedModel.problem_name==problem_name,
                    LearnedModel.k_fold==None))\
        .order_by(desc(LearnedModel.id)).limit(2).all()
    keeped_models = {m.id:m for m in last_two}

    top_models =  app.db.session.query(LearnedModel).\
        filter(and_(LearnedModel.k_fold==None,
                    LearnedModel.problem_name==problem_name)).\
        join(AccuracyMetric).order_by(desc(AccuracyMetric.accuracy)).all()

    for model in top_models:
        if len(keeped_models) >= keep_top_cnt + 2:
            break
        if model.id not in keeped_models:
            keeped_models[model.id] = model

    deleted = 0
    for model in top_models:
        if model.id not in keeped_models:
            app.db.session.delete(model)
            deleted += 1

    print('{} old models deleted'.format(deleted))
    app.db.session.flush()
    app.db.session.commit()

@app.celery.task(bind=True)
def run_test(self, taskname, update_errors=True):
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
            updater.finish(arrow.utcnow())
            return dump_result()

        updater.update_state(state='PROGRESS', progress=0.0, status='Waiting available gpu..')
        gpu_id = wait_available_gpu(self.request.id)

        last_metric_name = ''
        last_metric_value = 0.0

        for idx, gender_model in enumerate(gender_models):
            scale = 1.0 / num_models
            shift = idx * scale
            updater.push_rescale(shift, scale)
            updater.push_prefix('Model [{}/{}] '.format(idx + 1, num_models))

            run_gender_test(updater, gpu_id, gender_model)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing metrics..')
            metric_name, metric_value = compute_gender_metrics(gender_model)

            if idx == 0 and update_errors:
                updater.update_state(state='PROGRESS', progress=1.0,
                                     status='Computing errors..')
                compute_errors(gender_model)

                last_metric_name = metric_name
                last_metric_value = metric_value

            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='Successfully tested, {}={:.3f}'.format(metric_name, metric_value))

            updater.pop_prefix()
            updater.pop_rescale()

        updater.update_state(state='PROGRESS', progress=1.0, status='Releasing GPU..')
        release_gpu(self.request.id)

        updater.update_state(state='PROGRESS', progress=1.0,
                             status='Removing old models..')
        remove_old_main_models(taskname)

        updater.update_state(state='PROGRESS', progress=1.0,
                             status='Message to slack..')
        report_gender_metric()

        updater.update_state(state='SUCCESS', progress=1.0,
                             status='Successfully finished, {}={:.3f}'.format(last_metric_name, last_metric_value))

        updater.finish(arrow.utcnow())
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_test(updater, gpu_id, gender_model):

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing samples for testing..')
    samples = get_test_samples(gender_model.id, k_fold=gender_model.k_fold)
    print('number of samples: {}'.format(len(samples)))

    if len(samples) == 0:
        updater.update_state(state='PROGRESS', progress=1.0, status='No samples to test')
    else:
        # run training
        trainroom_dir = app.config.get('TRAINROOM_FOLDER')
        trainroom_gender = join(trainroom_dir, 'gender')

        snapshot = str(gender_model.prefix)
        epoch = gender_model.epoch
        exp_dir = str(gender_model.exp_dir)

        updater.push_rescale(0.01, 0.99)
        with add_path(trainroom_gender):
            test_module = __import__('test')
            metric_data, pr_probs = test_module.test(updater, snapshot, epoch, samples, exp_dir, gpu_id=gpu_id)
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
    ts = arrow.utcnow()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    release_gpu(uuid)

# train k-folds
@app.celery.task(bind=True)
def run_train_k_folds(self, taskname, k_fold):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        update_gender_cv_partition()
        run_gender_train(updater, self.request.id, k_fold=k_fold)
        updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')
        updater.finish(arrow.utcnow())
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
    ts = arrow.utcnow()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    gender_model = LearnedModel.query.filter_by(task_id=uuid)
    gender_model.delete()
    app.db.session.flush()
    app.db.session.commit()

    release_gpu(uuid)


def remove_old_k_folds_models(problem_name, except_id, k_fold):
    a = LearnedModel.query.filter(and_(LearnedModel.k_fold == k_fold,
                                       LearnedModel.id != except_id,
                                       LearnedModel.problem_name == problem_name)).delete()
    print('{} models deleted'.format(a))
    app.db.session.commit()

# test k-folds
@app.celery.task(bind=True)
def run_test_k_folds(self, taskname, k_fold):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        update_gender_cv_partition()

        gender_model = LearnedModel.query.filter_by(k_fold=k_fold, problem_name='gender').\
            order_by(desc(LearnedModel.id)).first()

        if gender_model:

            updater.update_state(state='PROGRESS', progress=0.0, status='Waiting available gpu..')
            gpu_id = wait_available_gpu(self.request.id)

            run_gender_test(updater, gpu_id, gender_model)

            updater.update_state(state='PROGRESS', progress=1.0, status='Releasing GPU..')
            release_gpu(self.request.id)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing metrics..')
            metric_name, metric_value = compute_gender_metrics(gender_model, update_state=False)

            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Computing errors..')
            compute_errors(gender_model)

            # remove old models
            updater.update_state(state='PROGRESS', progress=1.0,
                                 status='Removing old models..')
            remove_old_k_folds_models('gender', gender_model.id, k_fold)

            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='Successfully tested, {}={:.3f}'.format(metric_name, metric_value))
        else:
            updater.update_state(state='SUCCESS', progress=1.0,
                                 status='No model to test')
        updater.finish(arrow.utcnow())

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
    ts = arrow.utcnow()
    tasks.update(dict(finished_ts=ts, progress=1.0, state=state, status=status))
    app.db.session.flush()
    app.db.session.commit()

    release_gpu(uuid)

def wait_available_gpu(task_id):

    print('get gpu')
    gpu_ids = app.config.get('GPU_IDS')
    while True:
        selected_gpu = None
        with app.gpu_lock:
            status_data = app.db.session.query(GPUStatus).filter(GPUStatus.gpu_id.in_(gpu_ids)).all()
            status_data = {s.gpu_id:s for s in status_data}
            random.shuffle(gpu_ids)

            # get first random available gpu
            for gpu_id in gpu_ids:
                if gpu_id in status_data:
                    if not status_data[gpu_id].use:
                        status_data[gpu_id].use=True
                        status_data[gpu_id].task_id=task_id
                        selected_gpu = gpu_id
                        break
                else:
                    gpu_status = GPUStatus(gpu_id=gpu_id,task_id=task_id,use=True)
                    app.db.session.add(gpu_status)
                    selected_gpu = gpu_id
                    break

            app.db.session.flush()
            app.db.session.commit()

        if selected_gpu is None:
            print('not found, wait for 10 seconds..')
            time.sleep(10)
        else:
            break

    print('found available gpu: {}'.format(selected_gpu))

    return selected_gpu

def release_gpu(task_id):
    print('release gpu')
    with app.gpu_lock:
        a = GPUStatus.query.filter_by(task_id=task_id).update(dict(use=0))
        print('release {} gpus'.format(a))
        app.db.session.flush()
        app.db.session.commit()

    return True

def report_gender_metric():

    problem_name = 'gender'
    models = app.db.session.query(LearnedModel.id, AccuracyMetric.accuracy). \
        filter(and_(LearnedModel.k_fold == None,
                    LearnedModel.problem_name == problem_name,
                    LearnedModel.finished_ts != None)). \
        outerjoin(AccuracyMetric).order_by(desc(LearnedModel.id)).all()

    if len(models) == 0:
        return None

    models_tested = [m for m in models if m[1] is not None]
    if len(models_tested) == 0:
        return None

    best_model = max(models_tested, key=lambda x: x[1])

    cur_accuracy = models[0][1]
    if cur_accuracy is None:
        return None

    cur_error = 1 - cur_accuracy
    if len(models) > 1 and models[1][1] is not None:
        prev_accuracy = models[1][1]
        prev_error = 1 - prev_accuracy
        reduction = prev_error / cur_error if cur_error > 1e-12 else 1.0
        msg = 'Best model is #{}: accuracy: {:.3f}%\n'.format(best_model[0], 100 * best_model[1])
        msg += 'Previous model is #{}: accuracy: {:.3f}%\n'.format(models[1][0], 100 * prev_accuracy)
        msg += 'Last model is #{}: accuracy: {:.3f}%\n'.format(models[0][0], 100 * cur_accuracy)
        msg += '{}'.format(url_for('metrics', problem=problem_name))
    else:
        msg = 'Best model is #{}: accuracy: {:.3f}%\n'.format(best_model[0], 100 * best_model[1])
        msg += 'Last model is #{}: accuracy: {:.3f}%\n'.format(models[0][0], 100 * cur_accuracy)
        msg += '{}'.format(url_for('metrics', problem=problem_name))

    msg = ':loudspeaker: Test finished\n*Gender:*\n{}'.format(msg)
    send_slack_message(msg)

def report_gender_train(model):

    if model.finished_ts is None:
        return

    elapsed = model.finished_ts - model.started_ts

    msg = 'Model id: #{}\n'.format(model.id)
    msg += 'K-Fold: {}\n'.format(model.k_fold) if model.k_fold is not None else ''
    msg += 'Number of training samples: {}\n'.format(model.num_samples)
    msg += 'Training time: {}'.format(model.finished_ts.humanize(model.started_ts, only_distance=True))

    msg = ':loudspeaker: Train finished\n*Gender:*\n{}'.format(msg)
    send_slack_message(msg)