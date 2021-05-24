#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR
source ./buildImage.sh $1

$DOCKER_CMD push $DOCKER_SERVER/$DOCKER_IMAGE