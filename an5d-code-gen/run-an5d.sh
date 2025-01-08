#!/bin/bash

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $WORK_DIR

./rm-generated.sh

docker pull c506/an5d
docker run --user 1000:1000 -v $WORK_DIR:/mnt --rm c506/an5d -c "cd /mnt && /AN5D/an5d ./gol_32_an5d.cpp"
docker run --user 1000:1000 -v $WORK_DIR:/mnt --rm c506/an5d -c "cd /mnt && /AN5D/an5d ./gol_64_an5d.cpp"
