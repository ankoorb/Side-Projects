#!/bin/bash

PWD=$(pwd)
RUN_OPTS=""
RUN_OPTS+="-v $PWD/:/project "

IMAGE="nlp:latest"

CMD="docker run ${RUN_OPTS} -it ${IMAGE}"

echo ${CMD}

$CMD