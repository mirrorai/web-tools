echo 'SLEEP FOR 1 HOUR'
sleep 1h
python manage.py db_command -f data/db_add_detections.json
python manage.py check_detections