# -*- coding: utf-8 -*-

from . import app
from .script import CleanWasteImages, CleanCache


@app.celery.task()
def clean_images():
    print('clean images')

@app.celery.task()
def print_echo():
    print('echo from celery!')