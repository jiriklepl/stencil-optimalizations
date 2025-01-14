#!/bin/bash

WORK_DIR=$1
cd $WORK_DIR || exit 1

GOL_EXE_NAME="./stencils"

ALGORITHM="gol-cpu-naive"
GRID_DIMENSIONS_X="256"
GRID_DIMENSIONS_Y="256"
ITERATIONS="100"
DATA_LOADER_NAME="random-ones-zeros"
PATTERN_EXPRESSION=""
MEASURE_SPEEDUP="false"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-naive"
VALIDATE="false"
PRINT_VALIDATION_DIFF="false"
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

srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:V100 --time=2:00:00 $GOL_EXE_NAME \
    --algorithm="$ALGORITHM" \
    --grid-dimensions-x="$GRID_DIMENSIONS_X" \
    --grid-dimensions-y="$GRID_DIMENSIONS_Y" \
    --iterations="$ITERATIONS" \
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
