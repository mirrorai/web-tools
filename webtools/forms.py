# -*- coding: utf-8 -*-
from datetime import timedelta
import json
import re
import shapely.geometry

from wtforms import fields, Field, ValidationError
from wtforms_components import TimeRange
from wtforms_components.widgets import BaseDateTimeInput


class MoreThan(object):
    def __init__(self, other_field_name, message=None):
        self.other_field_name = other_field_name
        if not message:
            message = u'Field must be more than {}'.format(other_field_name)
        self.message = message

    def __call__(self, form, field):
        if field.data <= form[self.other_field_name].data:
            raise ValidationError(self.message)


# Inspired by https://www.snip2code.com/Snippet/719933/JSON-field-for-WTForms-that-converts-bet
class JsonField(fields.HiddenField):
    error_msg = 'This field contains invalid JSON'

    def _value(self):
        return json.dumps(self.data) if self.data else ''

    def process_formdata(self, formdata):
        if formdata:
            try:
                self.data = json.loads(formdata[0])
            except ValueError:
                raise ValueError(self.gettext(self.error_msg))
        else:
            # noinspection PyAttributeOutsideInit
            self.data = None


class ShapelyPolygonField(JsonField):
    error_msg = "Provided polygon isn't valid"

    def _value(self):
        # noinspection PyTypeChecker
        return json.dumps(list(self.data.exterior.coords)) if self.data else ''

    def process_formdata(self, formdata):
        super(ShapelyPolygonField, self).process_formdata(formdata)
        if self.data is not None:
            try:
                # noinspection PyAttributeOutsideInit
                self.data = shapely.geometry.Polygon(self.data)
                if not self.data.is_valid:
                    raise ValueError()
            except (ValueError, TypeError, IndexError):
                raise ValueError(self.gettext(self.error_msg))


# Inspired by https://github.com/kvesteri/wtforms-components/blob/master/wtforms_components/fields/time.py
class TimeDeltaInput(BaseDateTimeInput):
    input_type = 'time'
    range_validator_class = TimeRange
    format = '%H:%M:%S'

    def __init__(self, **kwargs):
        kwargs['step'] = kwargs.get('step', 1)
        super(TimeDeltaInput, self).__init__(**kwargs)


class TimeDeltaField(Field):
    """
    A text field which stores a `datetime.timedelta` matching a format.
    """
    widget = TimeDeltaInput()
    error_msg = 'Not a valid time interval. Provide it as HH:MM:SS or HH:MM'

    def _value(self):
        if self.raw_data:
            return ' '.join(self.raw_data)
        elif self.data is not None:
            return '{:02d}:{:02d}:{:02d}'.format(
                self.data.seconds / 3600,
                self.data.seconds % 3600 / 60,
                self.data.seconds % 60
            )
        else:
            return ''

    def process_formdata(self, value_list):
        if value_list:
            timedelta_str = ' '.join(value_list)
            try:
                regexp = r'\s*(?P<hours>\d)+\s*:\s*(?P<minutes>\d+)\s*(:\s*(?P<seconds>\d+)\s*)?'
                m = re.match(regexp, timedelta_str)

                if m is None:
                    raise ValueError(self.gettext(self.error_msg))

                hours = int(m.group('hours') or 0)
                minutes = int(m.group('minutes') or 0)
                seconds = int(m.group('seconds') or 0)

                if (hours + minutes + seconds) == 0:
                    raise ValueError(self.gettext("Time interval can't be zero"))

                self.data = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            except ValueError:
                # noinspection PyAttributeOutsideInit
                self.data = None
                raise ValueError(self.gettext(self.error_msg))
