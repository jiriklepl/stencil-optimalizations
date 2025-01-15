#!/bin/bash

WORK_DIR=$1
cd $WORK_DIR || exit 1

GOL_EXE_NAME="./stencils"

ALGORITHM="gol-cpu-bitwise-cols-naive-64"
GRID_DIMENSIONS_X="512"
GRID_DIMENSIONS_Y="1024"
ITERATIONS="100"

WARMUP_ROUNDS="0"
MEASUREMENT_ROUNDS="1"

DATA_LOADER_NAME="random-ones-zeros"
# DATA_LOADER_NAME="lexicon"
PATTERN_EXPRESSION="glider[10,10]"

MEASURE_SPEEDUP="true"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-naive"

VALIDATE="true"
PRINT_VALIDATION_DIFF="true"
VALIDATION_ALGORITHM_NAME="gol-cpu-naive"

ANIMATE_OUTPUT="false"
COLORFUL="true"

RANDOM_SEED="42"

THREAD_BLOCK_SIZE="0"

WARP_DIMS_X="0"
WARP_DIMS_Y="0"

WARP_TILE_DIMS_X="0"
WARP_TILE_DIMS_Y="0"

STREAMING_DIRECTION="naive"
MAX_RUNTIME_SECONDS="10"

# srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:V100 --time=2:00:00 $GOL_EXE_NAME \
srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:H100 --time=2:00:00 $GOL_EXE_NAME \
    --algorithm="$ALGORITHM" \
    --grid-dimensions-x="$GRID_DIMENSIONS_X" \
    --grid-dimensions-y="$GRID_DIMENSIONS_Y" \
    --iterations="$ITERATIONS" \
    --max-runtime-seconds="$MAX_RUNTIME_SECONDS" \
    --warmup-rounds="$WARMUP_ROUNDS" \
    --measurement-rounds="$MEASUREMENT_ROUNDS" \
    --data-loader="$DATA_LOADER_NAME" \
    --pattern-expression="$PATTERN_EXPRESSION" \
    --measure-speedup="$MEASURE_SPEEDUP" \
    --speedup-bench-algorithm="$SPEEDUP_BENCH_ALGORITHM_NAME" \
    --validate="$VALIDATE" \
    --print-validation-diff="$PRINT_VALIDATION_DIFF" \
    --validation-algorithm="$VALIDATION_ALGORITHM_NAME" \
    --animate-output="$ANIMATE_OUTPUT" \
    --colorful="$COLORFUL" \
    --random-seed="$RANDOM_SEED" \
    --thread-block-size="$THREAD_BLOCK_SIZE" \
    --warp-dims-x="$WARP_DIMS_X" \
    --warp-dims-y="$WARP_DIMS_Y" \
    --warp-tile-dims-x="$WARP_TILE_DIMS_X" \
    --warp-tile-dims-y="$WARP_TILE_DIMS_Y" \
    --streaming-direction="$STREAMING_DIRECTION"
