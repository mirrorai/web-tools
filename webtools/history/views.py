# -*- coding: utf-8 -*-
import shutil
import tempfile
import urllib
import zipfile

import arrow
import os
from datetime import timedelta
from flask import abort, flash, jsonify, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from flask_principal import Permission, RoleNeed
from flask_security import auth_token_required
from markupsafe import Markup
from sqlalchemy.orm import joinedload

from webtools import app
from webtools.user.forms import GrantAccessUserReferenceForm
from webtools.user.permissions import generate_permission_manipulating_endpoints
from webtools.utils import apply_min_max_detections_filters, apply_timing_filters, extract_archive, get_image, mkdir_p, \
    parse_min_max_detections_parameters, parse_timing_parameters, preprocess_paged_query, zipdir
from webtools.wrappers import nocache

# Shortcuts
db = app.db


@app.route('/history')
@login_required
def history():
    return render_template('history.html')

