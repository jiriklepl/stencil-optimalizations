#!/bin/bash

SCRIPT_TEMPLATE=$1
WORKER_ID=$2

echo "SCRIPT_TEMPLATE: $SCRIPT_TEMPLATE"
echo "WORKER_ID: $WORKER_ID"

SCRIPT="$SCRIPT_TEMPLATE--part_$WORKER_ID.sh"

echo "SCRIPT: $SCRIPT"
./$SCRIPT

# final tests
# ./final-measurements/_scripts/run-cpu-versions.sh


# hyper-params
# EXE="hyper-params-measurements/_scripts/search-cuda-naive-algs__part_$WORKER_ID.sh"
# ./$EXE

