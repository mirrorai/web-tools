#!/usr/bin/env bash
celery -A webtools.celery purge -f
celery -A webtools.celery worker -n celery@%h -B --concurrency 4 --loglevel=info -Q celery