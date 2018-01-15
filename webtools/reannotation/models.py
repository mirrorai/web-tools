# -*- coding: utf-8 -*-
import os
from flask_security import RoleMixin, UserMixin
from sqlalchemy_utils import IPAddressType, ArrowType

from webtools import app, opencv
from flask import send_file, abort
import webtools.utils as utils

from webtools.user.models import User

# Shortcuts
db = app.db

class ImagesDatabase(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    name = db.SDColumn(db.String(128))
    path = db.SDColumn(db.String(512), unique=True)

class Image(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    # Image properties
    imname = db.SDColumn(db.String(128))
    width = db.SDColumn(db.Integer)
    height = db.SDColumn(db.Integer)

    imdb_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('images_database.id', onupdate='CASCADE', ondelete='CASCADE')
    )
    imdb = db.relationship(
        ImagesDatabase,
        uselist=False,
        cascade='all, delete-orphan',
        single_parent=True
    )

    MAX_SIDE_SZ = 2096

    def send_image(self, **kwargs):
        image_file = self.filename()

        if not os.path.exists(image_file):
            app.logger.error(
                'Image %s for id %s is missing on disk. Returning 404...',
                image_file,
                self.id
            )
            abort(404)

        return Image._send_image(image_file, str(self.id), **kwargs)

    @staticmethod
    def _send_image(image_file, cache_name, minside=0, maxside=0):
        if minside > 0 or maxside > 0:

            minside = min(Image.MAX_SIDE_SZ, minside)
            maxside = min(Image.MAX_SIDE_SZ, maxside)

            utils.mkdir_p(app.config['IMAGE_CACHE_FOLDER'])
            resized_image_file_name = '{}_{}_{}.jpg'.format(cache_name, minside, maxside)
            cache_image_file = os.path.join(app.config['IMAGE_CACHE_FOLDER'], resized_image_file_name)
            if not os.path.exists(cache_image_file):
                im = opencv.read_image(image_file)
                im = opencv.resize_image(im, minside, maxside, avoid_upsampling=False)
                opencv.write_image(im, cache_image_file)

            image_file = cache_image_file

        return send_file(image_file, mimetype='image/jpeg')

    def filename(self):
        return self.imdb.path + '/' + self.imname

class TrainingTasks(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    task_name = db.SDColumn(db.String(64), unique=True)
    task_id = db.SDColumn(db.String(64))

# first iteration is created when user tries to annotate in the first time
# subsequently next iteration is created when training process is started
class GenderIteration(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    is_training = db.SDColumn(db.Boolean, default=False)

class GenderSample(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)

    image_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('image.id', onupdate='CASCADE', ondelete='CASCADE')
    )
    image = db.relationship(
        Image,
        uselist=False,
        cascade='all, delete-orphan',
        single_parent=True
    )

    # ground truth annotation
    is_annotated_gt = db.SDColumn(db.Boolean, default=False)
    is_male = db.SDColumn(db.Boolean, default=False)
    is_hard = db.SDColumn(db.Boolean, default=False)
    is_bad = db.SDColumn(db.Boolean, default=False)

    # is send for annotation, is checked already
    send_timestamp = db.SDColumn(ArrowType, nullable=True)
    is_checked = db.SDColumn(db.Boolean, default=False)

    # cv partition
    partition_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('gender_cross_valid_partition.id', onupdate='CASCADE', ondelete='CASCADE'),
        nullable=True
    )
    k_fold = db.SDColumn(db.Integer, nullable=True) # 0,1,2...K
    always_test = db.SDColumn(db.Integer, default=False) # always keep sample for testing

class GenderCrossValidPartition(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    k_folds = db.SDColumn(db.Integer) # number of folds

class GenderModel(db.Model):
    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    snapshot_prefix = db.SDColumn(db.String(512))
    epoch = db.SDColumn(db.Integer, nullable=True)
    k_fold = db.SDColumn(db.Integer, nullable=True)

class GenderUserIterAnnotation(db.Model):

    __table_args__ = (
        db.UniqueConstraint('sample_id', 'user_id', 'iteration_id'),
    )

    id = db.SDColumn(db.Integer, primary_key=True, autoincrement=True)
    sample_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('gender_sample.id', onupdate='CASCADE', ondelete='CASCADE')
    )
    sample = db.relationship(
        GenderSample,
        uselist=False,
        cascade='all, delete-orphan',
        single_parent=True
    )

    user_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('user.id', onupdate='CASCADE', ondelete='CASCADE')
    )
    user = db.relationship(
        User,
        uselist=False,
        cascade='all, delete-orphan',
        single_parent=True
    )

    iteration_id = db.SDColumn(
        db.Integer,
        db.ForeignKey('gender_iteration.id', onupdate='CASCADE', ondelete='CASCADE')
    )

    # annotation data
    is_male = db.SDColumn(db.Boolean)
    is_hard = db.SDColumn(db.Boolean, default=False)
    is_bad = db.SDColumn(db.Boolean, default=False)

    mark_timestamp = db.SDColumn(ArrowType)