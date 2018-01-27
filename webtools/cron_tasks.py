# -*- coding: utf-8 -*-

from . import app
from .script import CleanWasteModels, CleanCache, ResetSendSamples


@app.celery.task()
def clean_images():
    print('clean images')

@app.celery.task()
def print_echo():
    print('echo from celery!')

@app.celery.task()
def clean_waste_models():
    print('clean waste models')
    CleanWasteModels().run()

@app.celery.task()
def reset_send_samples():
    print('reset_send_samples')
    ResetSendSamples().run()

@app.celery.task()
def auto_training():
    print('auto training')
