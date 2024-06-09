#!/bin/bash

cd ./web_app

echo "Applying database migrations..."
python manage.py migrate

echo "Collecting static..."
python manage.py collectstatic --noinput

echo "Starting Django application..."
export RUNNING_SERVER=true
gunicorn -c gunicorn_config.py main.wsgi:application