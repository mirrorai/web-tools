from flask import current_app

from flask_testing import TestCase
from flask_migrate import Migrate

from webtools import app
from webtools.script import ResetDb


class FlaskTest(TestCase):
    def create_app(self):
        return app

    def setUp(self):
        Migrate(current_app, current_app.db)
        self.assertTrue(
            current_app.config['TESTING'],
            'Testing is not set. Are you sure you are using the right config?'
        )
        current_app.config['WTF_CSRF_ENABLED'] = False
        ResetDb.reset()
        self.client = self.app.test_client()

    def tearDown(self):
        ResetDb.reset()
