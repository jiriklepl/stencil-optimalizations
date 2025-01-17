#!/bin/bash

SCRIPT_TEMPLATE=$1
WORKER_ID=$2

echo "SCRIPT_TEMPLATE: $SCRIPT_TEMPLATE"
echo "WORKER_ID: $WORKER_ID"

SCRIPT="$SCRIPT_TEMPLATE--part_$WORKER_ID.sh"

echo "SCRIPT: $SCRIPT"
chmod +x $SCRIPT


EXECUTABLE='./internal-scripts/run-one-exp.sh' \
./$SCRIPT
