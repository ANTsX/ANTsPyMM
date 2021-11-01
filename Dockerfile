FROM python:3.7-slim-buster
LABEL maintainer="stnava"

RUN apt-get update && \
    apt-get install -y build-essential cmake libpng-dev pkg-config git

RUN pip install numpy keras boto3
RUN pip install --upgrade tensorflow tensorflow-probability

ARG antspy_hash
RUN pip install git+https://github.com/ANTsX/ANTsPy.git@$antspy_hash

ARG antspynet_hash
RUN pip install git+https://github.com/ANTsX/ANTsPyNet.git@$antspynet_hash

COPY setup.py src/setup.py
COPY superiq src/superiq
WORKDIR src
RUN python setup.py install
