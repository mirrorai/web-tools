#!/usr/bin/env bash
celery -A webtools.celery purge -f
celery -A webtools.celery worker -n learning@%h -B --concurrency 1 --loglevel=info -Q learning