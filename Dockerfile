FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
RUN pip install transformers peft --no-cache-dir
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]
