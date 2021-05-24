#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function error_exit {
  echo
  echo "$@"
  exit 1
}
trap "error_exit 'Received signal SIGHUP'" SIGHUP
trap "error_exit 'Received signal SIGINT'" SIGINT
trap "error_exit 'Received signal SIGTERM'" SIGTERM
trap "error_exit 'Error happened, failed to build image'" ERR
shopt -s expand_aliases
alias die='error_exit "Error ${0}(@`echo $(( $LINENO - 1 ))`):"'

# if the os is Linux, use sudo for docker
if [[ `uname` == 'Linux' ]]; then
  DOCKER_CMD="sudo docker"
else
  # if the os is Mac/Windows, don't use sudo since boot2docker needs some environment settings
  DOCKER_CMD="docker"
fi

hash docker >/dev/null 2>&1 || die "!! docker not installed, cannot continue"

REPO_ROOT=`git rev-parse --show-toplevel`
DOCKER_NAME="dwv-server"
DOCKER_SERVER="registry.cn-hangzhou.aliyuncs.com/bale_image"
DOCKER_IMAGE=${DOCKER_NAME}:1.0

echo "## copying binaries..."
rsync -az $REPO_ROOT/ $DIR/tmp --exclude node_modules --exclude docker 

# run docker build
echo "## building docker image..."
cd $DIR && $DOCKER_CMD build --rm -t $DOCKER_SERVER/$DOCKER_IMAGE -f Dockerfile tmp

echo "## finished building docker image"
echo "##   to push, use the following commands:"
echo "##   $DOCKER_CMD push $DOCKER_SERVER/$DOCKER_IMAGE"
rm -rf tmp