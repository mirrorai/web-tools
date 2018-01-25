#!/usr/bin/env bash

set -e

# FOLDER=db_backup

# sudo /etc/init.d/mysql stop
# sudo cp -r ${FOLDER}/webtools /var/lib/mysql/
# sudo /etc/init.d/mysql start

FILENAME=backup_2017_03_02_11_35_54.sql
DB_DUMP_PATH=/home/local/database/backup/${FILENAME}
CREATE_DB_CMD="drop database if exists ${MYSQL_DBNAME}_backup; create database ${MYSQL_DBNAME}_backup character set utf8;"
mysql -u root -p${MYSQL_PASS} --execute="${CREATE_DB_CMD}"
mysql -u root -p${MYSQL_PASS} ${MYSQL_DBNAME}_backup < ${DB_DUMP_PATH}
echo "restored"
