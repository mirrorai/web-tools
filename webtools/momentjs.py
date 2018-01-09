# -*- coding:utf-8 -*-

from jinja2 import Markup


# FIXME: remove this and use Flask-MomentJS
class MomentJS(object):
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def render(self, format):
        return Markup(
            '<script>\ndocument.write(moment("{}").{});\n</script>'.format(
                self.timestamp.strftime('%Y-%m-%dT%H:%M:%S Z'),
                format
            )
        )

    def object(self):
        return Markup('moment.utc("{}")'.format(self.timestamp.strftime('%Y-%m-%dT%H:%M:%S Z')))

    def format(self, fmt):
        return self.render('format("%s")' % fmt)

    def calendar(self):
        return self.render('calendar()')

    def from_now(self):
        return self.render('fromNow()')

    def format_full(self):
        return self.format('L LTS')
