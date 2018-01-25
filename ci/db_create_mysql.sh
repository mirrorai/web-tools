mysql -u root -p${MYSQL_PASS} \
    --execute="drop database if exists ${MYSQL_DBNAME}; create database ${MYSQL_DBNAME} character set utf8;"