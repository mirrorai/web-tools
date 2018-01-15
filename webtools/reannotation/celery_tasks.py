# -*- coding: utf-8 -*-

from webtools import app
import time
import random
from os.path import join
from models import GenderIteration
from sqlalchemy import func, or_, and_, desc, not_
from models import GenderSample
from ..utils import add_path
import sys

@app.celery.task(bind=True)
def run_train(self, taskname):
    if taskname == 'gender':
        return run_gender_train(self)
    else:
        print('uknown taskname: {}'.format(taskname))
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
                'result': 0}

def run_gender_train(self):

    # prepare samples
    samples = GenderSample.query.filter_by(is_bad=False,
                                           always_test=False,
                                           is_annotated_gt=True,
                                           is_hard=False).all()
    samples = [(s.image.filename(), 1 if s.is_male else 0) for s in samples]
    samples = samples[:1000]

    samples_test = GenderSample.query.filter_by(is_bad=False,
                                                always_test=True,
                                                is_annotated_gt=True,
                                                is_hard=False).all()
    samples_test = [(s.image.filename(), 1 if s.is_male else 0) for s in samples_test]
    n_val = 1000
    if len(samples_test) > n_val:
        samples_val = random.sample(samples_test, n_val)
    else:
        samples_val = samples_test

    trainroom_dir = app.config.get('TRAINROOM_FOLDER')
    trainroom_gender = join(trainroom_dir, 'gender')

    with add_path(trainroom_gender):
        trainroom_dir = app.config['TRAINROOM_FOLDER']
        exp_dir = join(trainroom_gender, 'exps', 'exp1')
        solve_module = __import__('solve')
        solve_module.solve(self, samples, samples_val, trainroom_dir, exp_dir)
        del sys.modules['solve']

    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}

@app.celery.task
def train_on_error(uuid):
    print('error!!!!')
    result = run_train.AsyncResult(uuid)
    exc = result.get(propagate=False)
    print('run_train id={0} raised exception: {1!r}\n{2!r}'.format(
          uuid, exc, result.traceback))

@app.celery.task
def train_on_success(uuid):
    result = run_train.AsyncResult(uuid)
    print('run_train id={0} successfully finished'.format(uuid))

@app.celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}