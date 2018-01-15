#!/usr/bin/env bash

flake8 --filename=*.py . && ./manage.py run_tests
