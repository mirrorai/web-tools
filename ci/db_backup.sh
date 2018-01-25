#!/usr/bin/env bash

set -e

# https://habrahabr.ru/post/63394/
# http://www.thegeekstuff.com/2008/07/backup-and-restore-mysql-database-using-mysqlhotcopy/
# http://database.ittoolbox.com/groups/technical-functional/mysql-l/lock-and-unlock-entire-mysql-database-3453557

# FOLDER=db_backup

# sudo /etc/init.d/mysql start
# echo "USE webtools; FLUSH TABLES WITH READ LOCK;" | mysql -u root -pMug0aenoPhi5koh4
# sudo rm -rf ${FOLDER}
# sudo mkdir -p ${FOLDER}
# sudo /usr/bin/mysqlhotcopy webtools ./${FOLDER}
# echo "USE webtools; UNLOCK TABLES;" | mysql -u root -pMug0aenoPhi5koh4

# https://habrahabr.ru/post/105954/

TIME_NOW=$(date +"%Y_%m_%d_%H_%M_%S")
FOLDER=/home/local/database/backup
mkdir -p ${FOLDER}
DB_DUMP_PATH=${FOLDER}/backup_${TIME_NOW}.sql
mysqldump -u root -p${MYSQL_PASS} ${MYSQL_DBNAME} > ${DB_DUMP_PATH}
echo "saved to ${DB_DUMP_PATH}"
