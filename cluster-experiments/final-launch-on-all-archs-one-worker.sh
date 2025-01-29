#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TEMPLATE_NAME"
    exit 1
fi

TEMPLATE_NAME=$1

./launch-workers-on.sh V100 ./final-measurements/_scripts/$TEMPLATE_NAME 1
./launch-workers-on.sh A100 ./final-measurements/_scripts/$TEMPLATE_NAME 1
./launch-workers-on.sh H100 ./final-measurements/_scripts/$TEMPLATE_NAME 1
