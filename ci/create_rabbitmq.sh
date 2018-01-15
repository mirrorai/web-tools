#!/usr/bin/env bash
rabbitmqctl add_user $RABBITMQ_NAME $RABBITMQ_PASSWORD
rabbitmqctl add_vhost $RABBITMQ_VHOST
rabbitmqctl set_permissions -p $RABBITMQ_VHOST $RABBITMQ_NAME ".*" ".*" ".*"