# -*- coding: utf-8 -*-

from webtools.user.permissions import create_permission_to_resource_instance, create_permission_to_resource_type
from .models import Camera

# noinspection PyTypeChecker
CameraReadNeed, get_cameras_read_base_query = create_permission_to_resource_instance(Camera, 'read')
# noinspection PyTypeChecker
CameraUpdateNeed, get_cameras_update_base_query = create_permission_to_resource_instance(Camera, 'update')
# noinspection PyTypeChecker
CameraManageNeed, get_cameras_manage_base_query = create_permission_to_resource_instance(Camera, 'manage')

# noinspection PyTypeChecker
CameraCreateNeed = create_permission_to_resource_type(Camera, 'create')
