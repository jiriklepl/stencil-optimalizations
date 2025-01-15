#!/bin/bash

#SBATCH -p gpu-short
#SBATCH -A kdss
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gres=gpu:V100
#SBATCH --time=2:00:00
#SBATCH --output=experiments-outputs/job-%j.out

echo "Starting job $SLURM_JOB_ID"

EXECUTABLE=../build/src/stencils

$EXECUTABLE \
    --algorithm="$ALGORITHM" \
    --grid-dimensions-x="$GRID_DIMENSIONS_X" \
    --grid-dimensions-y="$GRID_DIMENSIONS_Y" \
    --iterations="$ITERATIONS" \
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