# encoding: utf-8
from __future__ import unicode_literals
import glob
import json
import shutil
import os
import os.path
import re
import arrow
import opencv, cv2
from datetime import timedelta
from flask_migrate import downgrade, upgrade
from flask_script import Command, Option
from flask_security.confirmable import confirm_user
import numpy as np
import codecs

from os.path import join, splitext, basename, dirname, isfile, isdir

from . import app
from .reannotation.models import Image, GenderSample, UserGenderAnnotation, ImagesDatabase

from .utils import camelcase_to_snakecase, mkdir_p, list_subdirs, \
    list_images, check_image, validate_size, query_yes_no, get_number, find_fp_fn, \
    list_marking_files, load_xml, list_jsons, remove_simular, filehash

class ResetDb(Command):
    """
    Drops and recreate tables by downgrading with alembic to base and upgrade.
    Also removes image folders

    Note: Ensure that Migrate(app, app.db) has been called
    """
    def run(self, **kwargs):

        res = query_yes_no("Do you want to remove ALL data from database?")
        if not res:
            print('Canceled.')
            return

        self.reset()

    @staticmethod
    def reset():
        shutil.rmtree(app.config['IMAGE_FOLDER'])
        shutil.rmtree(app.config['IMAGE_CACHE_FOLDER'])
        downgrade(revision='base')
        upgrade()

class AddImageDB(Command):
    """Adds images to DB"""
    option_list = (
        Option(
            '--input_file', '-i',
            dest='input_file',
            help='Path to file with samples.'
        ),
        Option(
            '--dbname', '-n',
            dest='db_name',
            help='Database name.'
        ),
        Option(
            '--type', '-t',
            dest='db_type',
            help='Database type (\'gender\').'
        )
    )

    def __init__(self, func=None):
        super(AddImageDB, self).__init__(func=func)

    def load_samples(self, input_file):
        with open(input_file) as fp:
            content = fp.read().splitlines()

        samples = []
        for line in content:
            parts = line.split(';')
            local_path = parts[0]
            label = parts[1]
            samples.append((local_path, label))

        return samples

    def run(self, input_file, db_name, db_type):

        assert(db_type in ['gender'])

        samples = self.load_samples(input_file)

        base_dir = dirname(input_file)

        accepted_samples = []
        skipped = 0
        for local_path, label in samples:
            img_path = join(base_dir, local_path)
            if not isfile(img_path):
                skipped += 1
                continue
            img = cv2.imread(img_path, 1)
            if img is None:
                skipped += 1
                continue

            if label == 'f':
                is_male = False
            elif label == 'm':
                is_male = True
            else:
                print('wrong label: {}'.format(label))
                return

            accepted_samples.append((local_path, is_male, img.shape))


        print('total images accepted: {}'.format(len(accepted_samples)))
        print('total images skipped: {}'.format(skipped))

        print('adding data to database...')
        imdb = ImagesDatabase(name=db_name, path=base_dir)
        app.db.session.add(imdb)
        app.db.session.flush()

        for local_path, is_male, shape in accepted_samples:

            width = shape[1]
            height = shape[0]

            # create image instance
            image = Image(width=width, height=height, imname=local_path, imdb_id=imdb.id)
            app.db.session.add(image)
            app.db.session.flush()

            # create sample
            sample = GenderSample(image_id=image.id, is_male=is_male)
            # add to db
            app.db.session.add(sample)
            app.db.session.flush()

        # commit
        app.user_datastore.commit()
        app.db.session.commit()
        print('all done.')

class DatabaseCommand(Command):
    """Adds images to DB"""

    option_list = (
        Option(
            '--cmd_file', '-f',
            dest='cmd_file',
            help='Path to json file with commands.'
        ),
    )

    def __init__(self, func=None):
        super(DatabaseCommand, self).__init__(func=func)
        self.commands = None
        self.TEST_RATIO = 0.2

    # noinspection PyMethodOverriding
    def run(self, cmd_file):
        self.basedir = os.path.join('.')
        self.reldir = os.path.dirname(cmd_file)
        content = json.load(open(cmd_file))
        self.commands = content['commands']
        self.execute_commands()

    def execute_commands(self):

        for cmd in self.commands:
            cmd_name = cmd['name']
            if cmd_name == 'add_users':
                res = self.add_users(cmd)
            elif cmd_name == 'add_roles':
                res = self.add_roles(cmd)
            else:
                print('uknown command: {}'.format(cmd_name))
                return

            if not res:
                return

        # commit
        app.user_datastore.commit()
        app.db.session.commit()

    def add_roles(self, info):
        roles = info['roles']
        for role in roles:
            app.user_datastore.create_role(**role)
        app.user_datastore.commit()
        print('roles: {}'.format(len(roles)))
        return True

    def add_users(self, info):

        users = info['users']
        total = 0
        for user_info in users:
            build_info = user_info
            for i in range(user_info.get('_meta_multiply_object', 1)):
                user = app.user_datastore.create_user(**build_info)
                confirm_user(user)
                total += 1

        print('users: {}'.format(total))
        return True

class CleanWasteImages(Command):
    """Removes all snapshots and cached images that are not referenced from DB"""

    def run(self, **kwargs):
        self.clean_images()

    @staticmethod
    def clean_images():

        effective_ids = set()
        for image in Image.query.yield_per(100):
            effective_ids.add(image.id)

        removed_count = 0
        for image_file in glob.glob('{}/*/*'.format(app.config['IMAGE_FOLDER'])):
            try:
                image_id = long(image_file.split('/')[-1].split('.')[0])
            except ValueError:  # Skip unrecognized files
                pass
            if image_id in effective_ids:
                continue

            removed_count += 1
            os.remove(image_file)
            for cached_file in glob.glob('{}/{}_*'.format(app.config['IMAGE_CACHE_FOLDER'], image_id)):
                os.remove(cached_file)

        status_file = open(os.path.join(app.config['IMAGE_FOLDER'], 'clean_images_status.txt'), 'w')
        msg = '{} Total images removed: {}'.format(arrow.utcnow(), removed_count)
        print >> status_file, msg
        print(msg)

        status_file.close()


class CleanSamples(Command):
    """Removes all snapshots and cached images that are not referenced from DB"""

    def run(self, **kwargs):
        self.clean_samples()

    @staticmethod
    def clean_samples():

        res = query_yes_no("Do you want to remove all samples from DB?")
        if not res:
            print('Canceled.')
            return

        samples_count = GenderSample.query.count()

        UserGenderAnnotation.delete()
        ImagesDatabase.query.delete()
        GenderSample.query.delete()
        Image.query.delete()

        app.db.session.commit()

        shutil.rmtree(app.config['IMAGE_FOLDER'])
        shutil.rmtree(app.config['IMAGE_CACHE_FOLDER'])

        print('samples deleted: {}'.format(samples_count))

class CleanCache(Command):
    """Removes all cached images from DB"""

    def run(self, **kwargs):
        self.clean_cache_images()

    @staticmethod
    def clean_cache_images():

        removed_count = 0
        for image_file in glob.glob('{}/*/*'.format(app.config['IMAGE_FOLDER'])):
            try:
                image_id = long(image_file.split('/')[-1].split('.')[0])
            except ValueError:  # Skip unrecognized files
                pass

            for cached_file in glob.glob('{}/{}_*'.format(app.config['IMAGE_CACHE_FOLDER'], image_id)):
                os.remove(cached_file)
                removed_count += 1

        msg = '{} Total cache images removed: {}'.format(arrow.utcnow(), removed_count)
        print(msg)
