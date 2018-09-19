ARG BASE_IMAGE=ubuntu:16.04

FROM $BASE_IMAGE

ARG MODE=cpu

RUN apt-get update -y

RUN apt-get install -y libopencv-dev libcap-dev libgtk2.0

## Python3 Dependencies
RUN apt-get install -y python3 python3-dev python3-pip python3-setuptools python3-tk
RUN apt-get install -y swig

ADD files/ /

RUN pip3 install --no-cache-dir -r /requirements.txt
RUN pip3 uninstall -y tensorflow
RUN pip3 install --no-cache-dir -r /requirements-$MODE.txt

WORKDIR /playground

RUN apt-get clean
