#!/usr/bin/env bash
#celery -A webtools.celery -Q celery purge -f
celery -A webtools.celery worker -n celery@%h -B --concurrency 1 --loglevel=info -Q celery