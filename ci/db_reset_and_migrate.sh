#!/usr/bin/env bash
rm -f migrations/versions/*.py
rm -f migrations/versions/*.pyc
python manage.py db migrate
python manage.py db upgrade