FROM python:3.8.6
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python3 setup.py
CMD gunicorn main:app -w 1 -b 0.0.0.0:8080

