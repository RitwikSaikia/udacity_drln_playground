#!/bin/bash

IMAGE_NAME=drlnd_playground

extra_args=""

MODE=cpu
BASE_IMAGE=ubuntu:16.04
DOCKER=docker

while test $# -gt 0
do
    #Converting arg=$1 to lower case to make args case insensitive
    arg="$(tr [A-Z] [a-z] <<< "$1")"
    echo $arg
    case $arg in
    cpu)
        MODE=cpu
        BASE_IMAGE=ubuntu:16.04
        DOCKER=docker
        ;;
    gpu)
        MODE=gpu
        BASE_IMAGE=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
        DOCKER=nvidia-docker
        ;;
    *)
        extra_args="$extra_args $1"
    esac
    shift
done


set -x
exec $DOCKER build \
           --build-arg MODE=$MODE \
           --build-arg BASE_IMAGE=$BASE_IMAGE \
           $extra_args \
           -t ${IMAGE_NAME}:$MODE .

