#!/usr/bin/env python

from flask_script import Manager
from flask_script.commands import Server, Shell, ShowUrls, Clean
from flask_security.script import (CreateUserCommand, AddRoleCommand, RemoveRoleCommand, ActivateUserCommand,
                                   DeactivateUserCommand)
from flask_migrate import Migrate, MigrateCommand

from webtools import app, models
from webtools.script import ResetDb, DatabaseCommand, CleanWasteImages, CleanCache, \
        CleanSamples, AddImageDB
from webtools.tests.script import RunTests


def _make_context():
    return dict(app=app, db=app.db, models=models)


if __name__ == '__main__':
    migrate = Migrate(app, app.db)
    manager = Manager(app, with_default_commands=False)

    manager.add_command('shell', Shell(make_context=_make_context))
    manager.add_command('run_dev_server', Server(use_reloader=True, threaded=True))
    manager.add_command('show_urls', ShowUrls())
    manager.add_command('clean_pyc', Clean())

    manager.add_command('db', MigrateCommand)
    manager.add_command('db_reset', ResetDb)
    manager.add_command('db_command', DatabaseCommand)
    manager.add_command('add_imdb', AddImageDB)

    manager.add_command('clean_waste_images', CleanWasteImages)
    manager.add_command('clean_cache', CleanCache)
    manager.add_command('clean_samples', CleanSamples)

    manager.add_command('create_user', CreateUserCommand)
    manager.add_command('add_role', AddRoleCommand)
    manager.add_command('remove_role', RemoveRoleCommand)
    manager.add_command('deactivate_user', DeactivateUserCommand)
    manager.add_command('activate_user', ActivateUserCommand)

    manager.add_command('run_tests', RunTests())

    manager.run()
