FROM python:3.8.6
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install torch
RUN python3 setup.py
CMD python3 main.py
