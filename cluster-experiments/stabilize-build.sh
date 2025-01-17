#!/bin/bash

echo "Stabilizing build..."

echo "Remove old stable implementation..."
rm ../build-stable -rf

echo "Creating copy from build..."
cp ../build ../build-stable -r 

echo "Stabilizing build... Done!"