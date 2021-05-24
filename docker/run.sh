#!/bin/bash
if [[ `uname` == 'Linux' ]]; then
  DOCKER_CMD="sudo docker"
else
  # if the os is Mac/Windows, don't use sudo since boot2docker needs some environment settings
  DOCKER_CMD="docker"
fi

ContainerName="dwv-server"

$DOCKER_CMD pull registry.cn-hangzhou.aliyuncs.com/bale_image/dwv-server:1.0

exist=`docker inspect --format '{{.State.Running}}' ${ContainerName}`

if [ "${exist}" == "true" ]; then
    echo "${ContainerName} is running. Now stop and remove ${ContainerName}"
    $DOCKER_CMD stop $ContainerName && $DOCKER_CMD rm $ContainerName
else
    echo "Run a new ${ContainerName}"
fi

$DOCKER_CMD run -p 7777:7777 \
        -d \
        --name dwv-server \
        --rm \
        registry.cn-hangzhou.aliyuncs.com/bale_image/dwv-server:1.0