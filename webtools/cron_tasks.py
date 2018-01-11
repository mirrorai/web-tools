# -*- coding: utf-8 -*-

from . import app
from .script import CleanWasteImages, CleanCache


@app.celery.task()
def clean_images():
    CleanWasteImages().run()
    CleanCache().run()
