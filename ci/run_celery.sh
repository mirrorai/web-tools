#!/usr/bin/env bash
celery -A webtools.celery worker -B --concurrency 1