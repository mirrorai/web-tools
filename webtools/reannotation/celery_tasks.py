# -*- coding: utf-8 -*-

from webtools import app
import time
import random
from os.path import join, isdir
from os import mkdir, makedirs
from sqlalchemy import func, or_, and_, desc, not_
from models import GenderSample, LearningTask, LearnedModel, GenderUserAnnotation, GenderSampleResult
from shutil import copyfile
from ..utils import add_path
import sys
import arrow

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
            progress_n = (progress_n + shift) * scale
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

def get_samples(test=False, k_fold=None):
    samples_gt = app.db.session.query(GenderSample). \
        filter(and_(GenderSample.is_hard==False,
                    GenderSample.is_bad==False,
                    GenderSample.is_annotated_gt,
                    GenderSample.always_test==test)). \
        outerjoin(GenderUserAnnotation). \
        filter(GenderUserAnnotation.id == None).order_by(func.random()).limit(1000).all()

    samples_ann = app.db.session.query(GenderSample). \
        filter(GenderSample.always_test==test). \
        join(GenderUserAnnotation). \
        filter(and_(GenderUserAnnotation.is_hard==False,
                    GenderUserAnnotation.is_bad==False)). \
        order_by(func.random()).limit(1000).all()

    samples = [(s.image.filename(), 1 if s.is_male else 0) for s in samples_gt]
    samples.extend([(s.image.filename(), 1 if s.is_male else 0) for s in samples_ann])
    return samples

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

# train
@app.celery.task(bind=True)
def run_train(self, taskname):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        run_gender_train(updater, self.request.id)
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_train(updater, task_id, k_fold=None):

    updater.update_state(state='PROGRESS', progress=0.005, status='Preparing samples for training..')

    samples = get_samples()
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
    exp_fold = 'fold{}'.format(k_fold) if k_fold else 'main'
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
    model.finished_ts = arrow.now()
    app.db.session.flush()
    app.db.session.commit()

    updater.pop_rescale()
    updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')
    updater.finish(arrow.now())

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
@app.celery.task(bind=True)
def run_test(self, taskname):
    if taskname == 'gender':
        updater = StatusUpdater(self.request.id)
        run_gender_test(updater, self.request.id)
        return dump_result()
    else:
        print('uknown taskname: {}'.format(taskname))
        return dump_result()

def run_gender_test(updater, task_id, k_fold=None):

    updater.update_state(state='PROGRESS', progress=0.005, status='Preparing samples for testing..')
    samples = get_samples(test=True)

    if len(samples) == 0:
        updater.update_state(state='FAILURE', progress=1.0, status='No samples to test')
        updater.finish(arrow.now())
        return None

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing models for testing..')
    gender_model = LearnedModel.query.\
        filter(and_(LearnedModel.problem_name=='gender',LearnedModel.prefix!=None)). \
        order_by(desc(LearnedModel.id)).first()

    if not gender_model:
        updater.update_state(state='FAILURE', progress=1.0, status='No models to test')
        updater.finish(arrow.now())
        return None

    # run training
    trainroom_dir = app.config.get('TRAINROOM_FOLDER')
    trainroom_gender = join(trainroom_dir, 'gender')

    snapshot = str(gender_model.prefix)
    epoch = gender_model.epoch
    exp_dir = str(gender_model.exp_dir)

    updater.push_rescale(0.01, 0.99)
    updater.push_prefix('Model [1] ')
    with add_path(trainroom_gender):
        test_module = __import__('test')
        metric_data, pr_probs = test_module.test(updater, snapshot, epoch, samples, exp_dir)
        del sys.modules['test']

    updater.pop_rescale()
    updater.pop_prefix()

    metric_name, metric_value = metric_data

    assert(len(samples) == pr_probs.shape[0])
    for i in range(len(samples)):
        pr_prob = pr_probs[i,:]
        prob_neg = pr_prob[0]
        prob_pos = pr_prob[1]
        result_sample = GenderSampleResult(model_id=gender_model.id, prob_neg=prob_neg, prob_pos=prob_pos)
        app.db.session.add(result_sample)
        app.db.session.flush()

    app.db.session.commit()

    updater.update_state(state='SUCCESS', progress=1.0,
                         status='Models successfully tested, {}={:.3f}'.format(metric_name, metric_value))
    updater.finish(arrow.now())

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
        updater.push_rescale(0.0, 0.9)

        model_id = run_gender_train(updater, self.request.id, k_fold=k_fold)

        updater.pop_rescale()

        updater.push_rescale(0.9, 0.1)

        run_gender_test(updater, self.request.id, model_id=model_id)

        updater.pop_rescale()

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
        run_gender_test(updater, self.request.id, k_fold=k_fold)
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