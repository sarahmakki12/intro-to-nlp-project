FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

COPY requirements.txt /job/requirements.txt
RUN pip install --no-cache-dir -r /job/requirements.txt
