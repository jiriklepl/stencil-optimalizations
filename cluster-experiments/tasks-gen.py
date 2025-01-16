import itertools

MID_COORDS_MACRO = '{mid_coordinates}'

SCRIPT_EXE = './run-one-exp.sh'

def expand_macros(line):
    def read(coord: str):
        x = line.split(f'GRID_DIMENSIONS_{coord.upper()}="')[1].split('"')[0]
        return int(x)
    
    return line.replace(MID_COORDS_MACRO, str(read('x') // 2) + ',' + str(read('y') // 2))
        

class BenchSetUpOneTestCase:
    SPEED_UP_AND_VALIDATION_OFF = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '
    SPEED_UP_AND_VALIDATION_ON = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '

    GENERAL_SETTINGS = ' ITERATIONS="100000" MAX_RUNTIME_SECONDS="10" RANDOM_SEED="42" WARMUP_ROUNDS="10" MEASUREMENT_ROUNDS="10" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '

    TEST_CASES = [
        # f' GRID_DIMENSIONS_X="512"   GRID_DIMENSIONS_Y="512"   {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
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
    SPEED_UP_AND_VALIDATION_OFF = BenchSetUpOneTestCase.SPEED_UP_AND_VALIDATION_OFF
    SPEED_UP_AND_VALIDATION_ON = BenchSetUpOneTestCase.SPEED_UP_AND_VALIDATION_ON
    GENERAL_SETTINGS = BenchSetUpOneTestCase.GENERAL_SETTINGS
    TEST_CASES = BenchSetUpOneTestCase.TEST_CASES[:3]
    DATA = BenchSetUpOneTestCase.DATA

class AlgList:
    ALG_NAMES_with_variants = [
        ['gol-cpu-naive', [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],
        
        ['gol-cpu-bitwise-cols-naive-16', None],
        ['gol-cpu-bitwise-cols-naive-32', None],
        ['gol-cpu-bitwise-cols-naive-64', None],

        ['gol-cpu-bitwise-cols-macro-16', None],
        ['gol-cpu-bitwise-cols-macro-32', None],
        ['gol-cpu-bitwise-cols-macro-64', None],
        
        # 'gol-cuda-naive',
        
        # 'gol-cuda-naive-bitwise-cols-16',
        # 'gol-cuda-naive-bitwise-cols-32',
        # 'gol-cuda-naive-bitwise-cols-64',
        
        # 'gol-cuda-naive-bitwise-no-macro-16',
        # 'gol-cuda-naive-bitwise-no-macro-32',
        # 'gol-cuda-naive-bitwise-no-macro-64',
        
        # 'gol-cuda-naive-local-16',
        # 'gol-cuda-naive-local-32',
        # 'gol-cuda-naive-local-64',

        # 'gol-cuda-naive-just-tiling-16',
        # 'gol-cuda-naive-just-tiling-32',
        # 'gol-cuda-naive-just-tiling-64',
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
    
    

def interleaf_lines_with_echos(lines: list[str]):
    res = []
    for i, line in enumerate(lines.split('\n')):
        res.append(f'echo "exp-{i}"')
        res.append(line)
        res.append('\n')

    return '\n'.join(res)

task_gen = TaskGen(BenchSetUpForCPUs)

res = task_gen.generate_for_alg_list_to_str(AlgList.ALG_NAMES_with_variants, SCRIPT_EXE)

res = interleaf_lines_with_echos(res)

print(res)