#!/bin/sh

uwsgi_python --socket 0.0.0.0:8000 --workers 4 --protocol=https -w wsgi
