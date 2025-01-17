#!/bin/bash

EXECUTABLE=./run-one-exp.sh


echo "exp-72"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-73"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-74"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-75"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-76"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-77"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-78"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-79"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="32"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-80"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="32"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-81"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-82"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-83"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-84"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-85"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-86"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-87"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-88"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-89"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-90"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-91"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-92"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-93"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-94"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-95"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-96"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="32"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-97"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-98"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-99"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-100"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-101"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-102"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-103"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-104"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-105"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-106"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-107"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE