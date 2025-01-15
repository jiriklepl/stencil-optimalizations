#!/bin/bash

WORK_DIR=$1
cd $WORK_DIR || exit 1

GOL_EXE_NAME="./stencils"

__0="1"
__1="2"
__2="4"
__3="8"
__4="16"
__5="32"
__6="64"
__7="128"
__8="256"
__9="512"
__10="1024"
__11="2048"
__12="4096"
__13="8192"
__14="16384"
__15="32768"
__16="65536"
__17="131072"
__18="262144"
__19="524288"
__20="1048576"
__21="2097152"
__22="4194304"
__23="8388608"
__24="16777216"
__25="33554432"
__26="67108864"
__27="134217728"
__28="268435456"
__29="536870912"
__30="1073741824"
__31="2147483648"

# ALGORITHM="gol-cpu-bitwise-cols-naive-64"
# ALGORITHM="gol-cpu-bitwise-cols-macro-64"
# ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"
ALGORITHM="an5d-cpu-64"
GRID_DIMENSIONS_X=$__12
GRID_DIMENSIONS_Y=$__12
ITERATIONS="10"

WARMUP_ROUNDS="0"
MEASUREMENT_ROUNDS="1"

DATA_LOADER_NAME="random-ones-zeros"
# DATA_LOADER_NAME="lexicon"
# PATTERN_EXPRESSION="blinker[10,10]"
PATTERN_EXPRESSION="glider[10,10]"

# MEASURE_SPEEDUP="true"
MEASURE_SPEEDUP="false"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-naive"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-bitwise-cols-naive-64"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive"

VALIDATE="true"
PRINT_VALIDATION_DIFF="false"
# VALIDATION_ALGORITHM_NAME="gol-cpu-naive"
VALIDATION_ALGORITHM_NAME="gol-cuda-naive"

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
