#!/usr/bin/env bash
source venv/webtools/bin/activate
export PATH=$PATH:/usr/local/sbin
#export STAGE_CONFIG=config.DevelopmentConfig
# export STAGE_CONFIG=config.DevelopmentConfig
export STAGE_CONFIG=config.ProductionConfig
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export C_FORCE_ROOT=1
export SECURITY_PASSWORD_SALT='db7eb99841a744758ad030feb5dbac2f'
export MAIL_USERNAME='10x15print@gmail.com'
export MAIL_PASSWORD='lenadanil'
export MYSQL_LOGIN='root'
# export MYSQL_PASS='Mug0aenoPhi5koh4'
export SECRET_KEY='h\xcc\xb3i\x97V\x16-+\xc2\xe1\xe8\xd0\x06\xee\x01f\xc9\xe8\x10\x14\x88\xc9\x80'
export RABBITMQ_NAME='mirror'
export RABBITMQ_PASSWORD='mirrordev'
export RABBITMQ_VHOST='mirrorhost'
export MYSQL_DBNAME='webtools'
export MYSQL_PASS='SnVzhx3a88RSuYAy'
export SLACK_API_TOKEN='xoxp-104602030726-166136665859-305412984661-90dab3ff58a77841e5cdfc4711035b75'