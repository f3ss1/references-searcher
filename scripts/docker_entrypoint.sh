#!/bin/bash

cd ./web_app

echo "Applying database migrations..."
python manage.py migrate

echo "Collecting static..."
python manage.py collectstatic --noinput

echo "Starting Django application..."
export RUNNING_SERVER=true
gunicorn --workers 4 --bind 0.0.0.0:8000 main.wsgi:application