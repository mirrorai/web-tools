# -*- coding: utf-8 -*-

from webtools import app
import time
import random
from os.path import join, isdir
from os import mkdir, makedirs
from sqlalchemy import func, or_, and_, desc, not_
from models import GenderSample, LearningTask, LearnedModel
from shutil import copyfile
from ..utils import add_path
import sys
import arrow

class StatusUpdater:
    def __init__(self, task_id):
        self.task = LearningTask.query.filter_by(task_id=task_id)
        self.shift = 0.0
        self.scale = 1.0

    def update_state(self, state='PENDING', progress=0.0, status='Pending..'):
        task = self.task.first()
        if task:
            task.state = state
            task.status = status
            task.progress = (progress + self.shift) * self.scale
            app.db.session.flush()
            app.db.session.commit()

    def set_rescaler(self, shift, scale):
        self.shift = shift
        self.scale = scale

@app.celery.task(bind=True)
def dump_task(self):
    return {'current': 100, 'total': 100, 'status': 'Task completed.',
            'result': 0}

@app.celery.task(bind=True)
def run_train(self, taskname, k_fold=None):
    if taskname == 'gender':
        return run_gender_train(self, k_fold)
    else:
        print('uknown taskname: {}'.format(taskname))
        return {'current': 100, 'total': 100, 'status': 'Task completed.',
                'result': 0}

def run_gender_train(self, k_fold=None):

    updater = StatusUpdater(self.request.id)
    updater.update_state(state='PROGRESS', progress=0.005, status='Preparing samples for training..')

    # prepare samples
    samples = GenderSample.query.filter_by(is_bad=False,
                                           always_test=False,
                                           is_annotated_gt=True,
                                           is_hard=False).order_by(func.random()).limit(1000).all()
    samples = [(s.image.filename(), 1 if s.is_male else 0) for s in samples]

    samples_val = GenderSample.query.filter_by(is_bad=False,
                                               always_test=True,
                                               is_annotated_gt=True,
                                               is_hard=False).order_by(func.random()).limit(1000).all()
    samples_val = [(s.image.filename(), 1 if s.is_male else 0) for s in samples_val]

    updater.update_state(state='PROGRESS', progress=0.01, status='Preparing training scripts..')
    updater.set_rescaler(0.01, 0.99)

    # create model
    gender_model = LearnedModel(task_id=self.request.id, problem_name='gender')
    app.db.session.add(gender_model)
    app.db.session.flush()
    exp_num = gender_model.id

    trainroom_dir = app.config.get('TRAINROOM_FOLDER')
    trainroom_gender = join(trainroom_dir, 'gender')

    # prepare experiment directory
    exp_fold = 'fold{}'.format(k_fold) if k_fold else 'main'
    exp_dir = join(trainroom_gender, 'exps', 'exp{}_{}'.format(exp_num, exp_fold))
    exp_base = join(trainroom_gender, 'exps', 'base_exp')

    # add model to db
    gender_model.exp_dir = exp_dir
    app.db.session.flush()
    app.db.session.commit()

    # copy scripts for training
    if not isdir(exp_dir):
        makedirs(exp_dir)
        copyfile(join(exp_base, 'config.yml'), join(exp_dir, 'config.yml'))
        copyfile(join(exp_base, 'get_model.py'), join(exp_dir, 'get_model.py'))
        copyfile(join(exp_base, 'get_optimizer_params.py'), join(exp_dir, 'get_optimizer_params.py'))

    # run training
    with add_path(trainroom_gender):
        solve_module = __import__('solve')
        snapshot_prefix, epoch = solve_module.solve(updater, samples, samples_val, trainroom_dir, exp_dir)
        del sys.modules['solve']

    # udpate model in db
    gender_model.snapshot_prefix = snapshot_prefix
    gender_model.epoch = epoch
    gender_model.exp_dir = exp_dir
    app.db.session.flush()
    app.db.session.commit()

    # update task from db
    tasks = LearningTask.query.filter_by(task_id=self.request.id)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0))
    app.db.session.flush()
    app.db.session.commit()

    updater.set_rescaler(0.0, 1.0)
    updater.update_state(state='SUCCESS', progress=1.0, status='Model successfully trained')

    return {'current': 100, 'total': 100, 'status': 'Model successfully trained'}

def clear_data_for_train_task(uuid):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0, state='FAILURE', status='Error occurred while training'))
    app.db.session.flush()
    app.db.session.commit()

    gender_model = LearnedModel.query.filter_by(task_id=uuid)
    if gender_model.count() == 0:
        print('{}: no models'.format(uuid))
    else:
        print('{}: deleting models from db'.format(uuid))

    gender_model.delete()
    app.db.session.flush()
    app.db.session.commit()

@app.celery.task
def train_on_error(uuid):
    clear_data_for_train_task(uuid)
    return {'current': 1, 'total': 1, 'status': 'Error during training occured',
                'result': 0}

@app.celery.task
def train_on_success(result):
    print('run_train successfully finished')

@app.celery.task(bind=True)
def run_test(self, taskname, k_fold=None):
    if taskname == 'gender':
        return run_gender_test(self, k_fold)
    else:
        print('uknown taskname: {}'.format(taskname))
        return {'current': 100, 'total': 100, 'status': 'Task completed.',
                'result': 0}

def run_gender_test(self, k_fold=None):

    self.update_state(state='PROGRESS',
                      meta={'current': 0, 'total': 1, 'status': 'Preparing samples for testing..'})

    time.sleep(5)

    for i in range(10):
        self.update_state(state='PROGRESS',
                          meta={'current': i + 1, 'total': 10, 'status': 'Testing..'})
        time.sleep(2)

    # remove task from db
    tasks = LearningTask.query.filter_by(task_id=self.request.id)
    if tasks.count() == 0:
        print('{}: no task'.format(self.request.id))
    else:
        print('{}: deleting task from db'.format(self.request.id))

    tasks.delete()
    app.db.session.flush()
    app.db.session.commit()

    return {'current': 100, 'total': 100, 'status': 'Testing successfully finished.'}

def clear_data_for_test_task(uuid):
    tasks = LearningTask.query.filter_by(task_id=uuid)
    ts = arrow.now()
    tasks.update(dict(finished_ts=ts, progress=1.0))
    app.db.session.flush()
    app.db.session.commit()

@app.celery.task
def test_on_error(uuid):
    clear_data_for_test_task(uuid)
    return {'current': 1, 'total': 1, 'status': 'Error during testing occured.',
            'result': 0}

@app.celery.task
def test_on_success(result):
    print('run_train successfully finished')