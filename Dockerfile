FROM python:3.11.0-slim as base

FROM base as builder
RUN apt-get update && apt-get install -y git
FROM base
# turn off python output buffering
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /home/exp
WORKDIR /home/exp
COPY ./src /home/exp/src
COPY ./requirements.txt /home/exp/requirements.txt
RUN pip install -r /home/exp/requirements.txt
EXPOSE 8000
ENTRYPOINT ["python", "/home/exp/src/run.py"]
