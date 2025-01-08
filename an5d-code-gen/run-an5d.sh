#!/bin/bash

SOURCE_KEY_FOR_32="gol_32_an5d"
SOURCE_KEY_FOR_64="gol_64_an5d"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $WORK_DIR

./rm-generated.sh



# Use the "sed" command to search for "kernel0_" and replace it with "_32_kernel0_"


docker pull c506/an5d

docker run --user 1000:1000 -v $WORK_DIR:/mnt --rm c506/an5d -c "cd /mnt && /AN5D/an5d ./${SOURCE_KEY_FOR_32}.cpp"
sed -i 's/kernel0_/_32_kernel0_/g' ./${SOURCE_KEY_FOR_32}*

docker run --user 1000:1000 -v $WORK_DIR:/mnt --rm c506/an5d -c "cd /mnt && /AN5D/an5d ./${SOURCE_KEY_FOR_64}.cpp"
sed -i 's/kernel0_/_64_kernel0_/g' ./${SOURCE_KEY_FOR_64}*
