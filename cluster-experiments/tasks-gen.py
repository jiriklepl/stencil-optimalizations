import itertools
import random

MID_COORDS_MACRO = '{mid_coordinates}'

SCRIPT_EXE = '$EXECUTABLE'

def expand_macros(line):
    def read(coord: str):
        x = line.split(f'GRID_DIMENSIONS_{coord.upper()}="')[1].split('"')[0]
        return int(x)
    
    return line.replace(MID_COORDS_MACRO, str(read('x') // 2) + ',' + str(read('y') // 2))
        

class BenchSetUp:
    SPEED_UP_AND_VALIDATION_OFF = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '
    SPEED_UP_AND_VALIDATION_ON = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '

    GENERAL_SETTINGS = ' ITERATIONS="100000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="10" MEASUREMENT_ROUNDS="10" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '

    TEST_CASES = [
        f' GRID_DIMENSIONS_X="1024"  GRID_DIMENSIONS_Y="1024"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="2048"  GRID_DIMENSIONS_Y="2048"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="4096"  GRID_DIMENSIONS_Y="4096"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
    ]

    DATA= [
        f'DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]" ',
    ]

class BenchSetUpForCPUs:
    SPEED_UP_AND_VALIDATION_OFF = BenchSetUp.SPEED_UP_AND_VALIDATION_OFF
    SPEED_UP_AND_VALIDATION_ON = BenchSetUp.SPEED_UP_AND_VALIDATION_ON
    GENERAL_SETTINGS = BenchSetUp.GENERAL_SETTINGS
    TEST_CASES = BenchSetUp.TEST_CASES[:6]
    DATA = BenchSetUp.DATA

class BenchSetUpForGPUS:
    SPEED_UP_AND_VALIDATION_OFF = BenchSetUp.SPEED_UP_AND_VALIDATION_OFF
    SPEED_UP_AND_VALIDATION_ON = BenchSetUp.SPEED_UP_AND_VALIDATION_ON

    GENERAL_SETTINGS = ' ITERATIONS="10000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="10" MEASUREMENT_ROUNDS="10" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '
    
    TEST_CASES = [
        f' GRID_DIMENSIONS_X="1024"  GRID_DIMENSIONS_Y="1024"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="2048"  GRID_DIMENSIONS_Y="2048"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="4096"  GRID_DIMENSIONS_Y="4096"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
    ]

    DATA = BenchSetUp.DATA

class AlgList:
    CPU_ALGS_with_variants = [
        ['gol-cpu-naive', [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],
        
        ['gol-cpu-bitwise-cols-naive-16', None],
        ['gol-cpu-bitwise-cols-naive-32', None],
        ['gol-cpu-bitwise-cols-naive-64', None],

        ['gol-cpu-bitwise-cols-macro-16', None],
        ['gol-cpu-bitwise-cols-macro-32', None],
        ['gol-cpu-bitwise-cols-macro-64', None],
    ]
    CUDA_NAIVE_ALGS_with_variants = [
        # CUDA NAIVE ALGORITHMS
            # Thread block sizes has been experimentally determined
            # note: the differences in performance have been minimal

        ['gol-cuda-naive', [' THREAD_BLOCK_SIZE="512" BASE_GRID_ENCODING="char" ', ' THREAD_BLOCK_SIZE="64" BASE_GRID_ENCODING="int"  ']],

        ['gol-cuda-naive-bitwise-cols-32',     [' THREAD_BLOCK_SIZE="512" ']],
        ['gol-cuda-naive-bitwise-cols-64',     [' THREAD_BLOCK_SIZE="256" ']],
        
        ['gol-cuda-naive-bitwise-no-macro-32', [' THREAD_BLOCK_SIZE="512" ']],
        ['gol-cuda-naive-bitwise-no-macro-64', [' THREAD_BLOCK_SIZE="256" ']],
 
        # 'gol-cuda-naive-local-16',
        # 'gol-cuda-naive-local-32',
        # 'gol-cuda-naive-local-64',
    ]

class HyperParamsCases:
    
    GENERAL_SETTINGS = ' ITERATIONS="1000000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="1" MEASUREMENT_ROUNDS="3" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '

    TEST_CASES = [
        f' GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"  {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536" {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
    ]

    TEST_CASES_small_set = [
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" {GENERAL_SETTINGS} {BenchSetUp.SPEED_UP_AND_VALIDATION_OFF} ',
    ]

    NAIVE_CUDA_ALGS_with_variants = [
        ['gol-cuda-naive',  [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],
        ['gol-cuda-naive-bitwise-cols-32', None],
        ['gol-cuda-naive-bitwise-cols-64', None],
        ['gol-cuda-naive-bitwise-no-macro-32', None],
        ['gol-cuda-naive-bitwise-no-macro-64', None],
    ]

    BLOCK_SIZES = [
        ' THREAD_BLOCK_SIZE="32" ',
        ' THREAD_BLOCK_SIZE="64" ',
        ' THREAD_BLOCK_SIZE="128" ',
        ' THREAD_BLOCK_SIZE="256" ',
        ' THREAD_BLOCK_SIZE="512" ',
        ' THREAD_BLOCK_SIZE="1024" ',
    ]

    LOCAL_ALGS_with_variants = [
        ['gol-cuda-naive-local-32', None],
        ['gol-cuda-naive-local-64', None],
    ]

    DATA = [
        # space filler
        f' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]"  '

        # no action
        ' DATA_LOADER_NAME="zeros" '

        # always changing
        ' DATA_LOADER_NAME="always-changing" '
    ]

    WARP_DIMS = [
        ' WARP_DIMS_X="1" WARP_DIMS_Y="32" ',
        ' WARP_DIMS_X="32" WARP_DIMS_Y="1" ',
        ' WARP_DIMS_X="16" WARP_DIMS_Y="2" ',
        ' WARP_DIMS_X="2" WARP_DIMS_Y="16" ',
        ' WARP_DIMS_X="8" WARP_DIMS_Y="4" ',
        ' WARP_DIMS_X="4" WARP_DIMS_Y="8" ',
    ]

    WARP_TILE_DIMS = [
        ' WARP_TILE_DIMS_X="32" WARP_TILE_DIMS_Y="32" ',
        ' WARP_TILE_DIMS_X="8" WARP_TILE_DIMS_Y="32" ',
        ' WARP_TILE_DIMS_X="32" WARP_TILE_DIMS_Y="8" ',
        ' WARP_TILE_DIMS_X="64" WARP_TILE_DIMS_Y="64" ',
        ' WARP_TILE_DIMS_X="32" WARP_TILE_DIMS_Y="64" ',
        ' WARP_TILE_DIMS_X="64" WARP_TILE_DIMS_Y="32" ',
        ' WARP_TILE_DIMS_X="128" WARP_TILE_DIMS_Y="128" ',
        ' WARP_TILE_DIMS_X="128" WARP_TILE_DIMS_Y="64" ',
        ' WARP_TILE_DIMS_X="64" WARP_TILE_DIMS_Y="128" ',
        ' WARP_TILE_DIMS_X="128" WARP_TILE_DIMS_Y="32" ',
        ' WARP_TILE_DIMS_X="32" WARP_TILE_DIMS_Y="128" ',
        ' WARP_TILE_DIMS_X="256" WARP_TILE_DIMS_Y="128" ',
        ' WARP_TILE_DIMS_X="128" WARP_TILE_DIMS_Y="256" ',
        ' WARP_TILE_DIMS_X="256" WARP_TILE_DIMS_Y="32" ',
        ' WARP_TILE_DIMS_X="256" WARP_TILE_DIMS_Y="64" ',
    ]

    STREAMING_DIRECTIONS = [
        ' STREAMING_DIRECTION="in-x" ',
        ' STREAMING_DIRECTION="in-y" ',
        ' STREAMING_DIRECTION="naive" ',
    ]

    STATE_BITS_COUNTS = [
        ' STATE_BITS_COUNT="32" ',
        ' STATE_BITS_COUNT="64" ',
    ]


class VariantGenerator:
    def __init__(self, prefix=''):
        self.prefix = prefix

    def generate(self, variants):
        for variant in itertools.product(*variants):
            yield variant

    def generate_to_arr_of_lines(self, variants):
        return [' '.join(variant) for variant in self.generate(variants)]

    def generate_to_str(self, variants):
        return '\n'.join(self.generate_to_arr_of_lines(variants))


class TaskGen:
    def __init__(self, SETTINGS):
        self.SETTINGS = SETTINGS
        
    def generate_for_alg(self, alg_keys_and_variants):
        
        alg_params = []

        for alg_key, alg_variant in alg_keys_and_variants:
            alg_variant = alg_variant if alg_variant is not None else ['']

            for variant in alg_variant:
                alg_params.append(f'ALGORITHM="{alg_key}" {variant}')

        variants = [
            self.SETTINGS.TEST_CASES,
            self.SETTINGS.DATA,

            alg_params,
        ]

        generator = VariantGenerator()
        variant_lines = generator.generate_to_arr_of_lines(variants)

        return [expand_macros(line) for line in variant_lines]
    
    def generate_for_alg_list_to_str(self, alg_list, suffix):
        lines_with_suffix = [f'{line} {suffix}' for line in self.generate_for_alg(alg_list)]

        return '\n'.join(lines_with_suffix)
    

class HyperParamsTaskGen:
    def __init__(self):
        self.TEST_CASES = HyperParamsCases.TEST_CASES
        self.DATA = BenchSetUp.DATA

    def set_test_cases(self, test_cases):
        self.TEST_CASES = test_cases
        return self
    
    def set_data(self, data):
        self.DATA = data
        return self

    def generate_for_alg(self, alg_keys_and_variants, hp_space):
        
        alg_params = []

        for alg_key, alg_variant in alg_keys_and_variants:
            alg_variant = alg_variant if alg_variant is not None else ['']

            for variant in alg_variant:
                alg_params.append(f'ALGORITHM="{alg_key}" {variant}')


        variants = [
            self.TEST_CASES,

            alg_params,
            *hp_space,
            
            self.DATA,
        ]

        generator = VariantGenerator()
        variant_lines = generator.generate_to_arr_of_lines(variants)
        variant_lines = self._shuffle_lines(variant_lines)

        return [expand_macros(line) for line in variant_lines]
    
    def generate_for_alg_list_to_str(self, alg_list, hp_space, suffix):
        lines_with_suffix = [f'{line} {suffix}' for line in self.generate_for_alg(alg_list, hp_space)]

        return '\n'.join(lines_with_suffix)

    def _shuffle_lines(self, lines: list[str]):
        for i in range(len(lines)):
            j = random.randint(0, len(lines) - 1)
            lines[i], lines[j] = lines[j], lines[i]
        return lines

def has_valid_warp_tile_dims(line):
    def read_val(key, line):
        return int(line.split(f'{key}="')[1].split('"')[0])
    
    x_tile = read_val('WARP_TILE_DIMS_X', line)
    y_tile = read_val('WARP_TILE_DIMS_Y', line)

    x_warp = read_val('WARP_DIMS_X', line)
    y_warp = read_val('WARP_DIMS_Y', line)

    warp_fits_tile_in_x = x_tile % x_warp == 0
    warp_fits_tile_in_y = y_tile % y_warp == 0

    return warp_fits_tile_in_x and warp_fits_tile_in_y

def interleaf_lines_with_echos(lines: list[str], start: int = 0):
    res = []
    for i, line in enumerate(lines.split('\n')):
        res.append(f'echo "exp-{i + start}"')
        res.append(line)
        res.append('\n')

    return '\n'.join(res)



# final cases generation
# res = TaskGen(BenchSetUpForCPUs).generate_for_alg_list_to_str(AlgList.CPU_ALGS_with_variants, SCRIPT_EXE)
res = TaskGen(BenchSetUpForGPUS).generate_for_alg_list_to_str(AlgList.CUDA_NAIVE_ALGS_with_variants, SCRIPT_EXE)
res = interleaf_lines_with_echos(res)
print(res)
exit()

# hyperparams cases generation - naive cuda
# res = HyperParamsTaskGen().generate_for_alg_list_to_str(
#     HyperParamsCases.NAIVE_CUDA_ALGS_with_variants, [HyperParamsCases.BLOCK_SIZES], SCRIPT_EXE)

# hyperparams cases generation - local alg
res = HyperParamsTaskGen() \
    .set_test_cases(HyperParamsCases.TEST_CASES_small_set) \
    .set_data(HyperParamsCases.DATA) \
    .generate_for_alg_list_to_str(
        HyperParamsCases.LOCAL_ALGS_with_variants, 
        [
            HyperParamsCases.WARP_DIMS,
            HyperParamsCases.WARP_TILE_DIMS,
            HyperParamsCases.STREAMING_DIRECTIONS,
            HyperParamsCases.STATE_BITS_COUNTS,
        ],
        SCRIPT_EXE)

res_lines = []
skipped = 0

for line in res.split('\n'):
    if has_valid_warp_tile_dims(line):
        res_lines.append(line)
    else:
        skipped += 1

print(f'Skipped: {skipped} lines due to invalid warp tile dims')

parts = 4
filenames = './hyper-params-measurements/_scripts/search-cuda-local--part_{i}.sh'
file_prefix = '#!/bin/bash\n\nEXECUTABLE=./run-one-exp.sh\n\n'

for i in range(parts):
    fname = filenames.replace('{i}', str(i + 1))
    print (f'Writing to {fname}')
    
    content = '\n'.join(res_lines[i::parts])
    content = interleaf_lines_with_echos(content, i * len(res_lines) // parts)
    
    with open(fname, 'w') as f:
        f.write(file_prefix)
        f.write(content)

