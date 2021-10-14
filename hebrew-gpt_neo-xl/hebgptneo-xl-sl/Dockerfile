FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest
ADD . /app
WORKDIR /app
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

RUN python3 setup.py

# remember to expose the port your app'll be exposed on.
EXPOSE 8080

# run it!
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
