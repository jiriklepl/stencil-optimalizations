import sys
from general_job_generation import *

class HPs:
    SPEED_UP_AND_VALIDATION_OFF = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '
    
    PRIMARY_SETTINGS = ' ITERATIONS="100000" MAX_RUNTIME_SECONDS="5" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '
    
    BLOCK_SIZES = [
        ' THREAD_BLOCK_SIZE="32" ',
        ' THREAD_BLOCK_SIZE="64" ',
        ' THREAD_BLOCK_SIZE="128" ',
        ' THREAD_BLOCK_SIZE="256" ',
        ' THREAD_BLOCK_SIZE="512" ',
        ' THREAD_BLOCK_SIZE="1024" ',
    ]
    VARIOUS_DATA_LOADERS = [
        f' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]" TAG="spacefiller"  ',
        ' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="gosper-glider-gun[0,0]" TAG="glider-gun"  ',
        ' DATA_LOADER_NAME="zeros" TAG="no-work" ',
        ' DATA_LOADER_NAME="always-changing" TAG="full-work" ',
    ]
    WHATEVER_DATA_LOADER = [
        f' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]"  ',
    ]
    STATE_BITS_COUNTS = [
        ' STATE_BITS_COUNT="32" ',
        ' STATE_BITS_COUNT="64" ',
    ]
    GRIDS = [
        ' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" ',
        # ' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" ',
    ]

simple_hps = [
    [HPs.PRIMARY_SETTINGS],
    [HPs.SPEED_UP_AND_VALIDATION_OFF],
    HPs.BLOCK_SIZES,
    HPs.WHATEVER_DATA_LOADER,
    HPs.GRIDS,
]

local_hps = [
    [HPs.PRIMARY_SETTINGS],
    [HPs.SPEED_UP_AND_VALIDATION_OFF],
    HPs.BLOCK_SIZES,
    HPs.VARIOUS_DATA_LOADERS,
    HPs.GRIDS,
    HPs.STATE_BITS_COUNTS,
]

per_alg_hps = [
    [['gol-cuda-naive', [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],
     simple_hps],

    [['gol-cuda-naive-bitwise-cols-32', None],
     simple_hps],
    
    [['gol-cuda-naive-bitwise-cols-64', None],
     simple_hps],

    [['gol-cuda-naive-bitwise-tiles-32', None],                                              
     simple_hps],
    
    [['gol-cuda-naive-bitwise-tiles-64', None],                                              
     simple_hps],

    [['gol-cuda-naive-bitwise-no-macro-32', None],                                           
     simple_hps],
    
    [['gol-cuda-naive-bitwise-no-macro-64', None],                                           
     simple_hps],

    [['gol-cuda-local-one-cell-cols-32', None],                                              
     local_hps],
    
    [['gol-cuda-local-one-cell-cols-64', None],                                              
     local_hps],

    [['gol-cuda-local-one-cell-32--bit-tiles', None],                                        
     local_hps],
    
    [['gol-cuda-local-one-cell-64--bit-tiles', None],                                        
     local_hps],

    # Related work

    [['eff-baseline', None],
     simple_hps],

    [['eff-baseline-shm', None],
     simple_hps],

    [['eff-baseline-texture', None],
     simple_hps],

    [['eff-sota-packed-32', None],
     simple_hps],

    [['eff-sota-packed-64', None],
     simple_hps],
]

if len(sys.argv) != 3:
    print('Usage: generate-hp-search.py <template_name> <workers_count>')
    sys.exit(1)

res = Generator() \
    .set_algs_and_hps(per_alg_hps) \
    .generate_all()

print ('# generated: ', len(res))

template_name = sys.argv[1]
workers_count = int(sys.argv[2])

folder = 'hyper-params-measurements'
write_to_files(folder, res, template_name, workers_count)
