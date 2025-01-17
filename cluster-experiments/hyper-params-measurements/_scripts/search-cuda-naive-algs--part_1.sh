#!/bin/bash

EXECUTABLE=./run-one-exp.sh
echo "exp-0"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-1"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-2"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-3"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-4"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-5"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-6"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-7"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-8"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-9"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-10"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-11"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-12"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-13"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-14"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-15"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-16"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-17"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-18"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-19"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="32"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-20"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-21"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-cols-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-22"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="32"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-23"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-24"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE


echo "exp-25"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-26"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="512"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-27"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="256"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-28"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-29"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-30"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="1024"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-31"
 GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[16384,16384]"  $EXECUTABLE


echo "exp-32"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive-bitwise-no-macro-32"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-33"
 GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="64"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[32768,32768]"  $EXECUTABLE


echo "exp-34"
 GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384"  ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="int"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[8192,8192]"  $EXECUTABLE


echo "exp-35"
 GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"   ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char"   MEASURE_SPEEDUP="false"  VALIDATE="false"   ALGORITHM="gol-cuda-naive"  BASE_GRID_ENCODING="char"   THREAD_BLOCK_SIZE="128"  DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096,4096]"  $EXECUTABLE

