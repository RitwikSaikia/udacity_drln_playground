#!/bin/bash

IMAGE_NAME=drlnd_playground

extra_args=""

MODE=cpu
DOCKER=docker

# Mapping of args to gradle build command
ARGS_MAPPING=("cpu", "gpu")

while test $# -gt 0
do
    #Converting arg=$1 to lower case to make args case insensitive
    arg="$(tr [A-Z] [a-z] <<< "$1")"
    echo $arg
    case $arg in
    cpu)
        MODE=cpu
        DOCKER=docker
        ;;
    gpu)
        MODE=gpu
        DOCKER=nvidia-docker
        ;;
    *)
        extra_args="$extra_args $1"
    esac
    shift
done

WORKSPACE=$(cd ../ && pwd)

set -x

exec $DOCKER run \
        -v $WORKSPACE:/playground \
        $extra_args \
        -it ${IMAGE_NAME}:$MODE /bin/bash

