#!/bin/bash

PWD=$(pwd)
RUN_OPTS=""
RUN_OPTS+=" -p 8888:8888 "
RUN_OPTS+="-v $PWD/:/project "

IMAGE="nlp:latest"

CMD="docker run ${RUN_OPTS} -it ${IMAGE} jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token=''"

echo ${CMD}

$CMD