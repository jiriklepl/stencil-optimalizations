import os
import random
import results_abstraction as ra
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = './final-measurements/hopper'
# BASE_DIR = './final-measurements/ampere'
# BASE_DIR = './experiments-outputs'
GRAPH_DIR = './generated-graphs'

MODE='png'
# MODE='pdf'

PLOT_NAME_template = '__tmp_{plot_type}.' + MODE

class LegendNames:
    @staticmethod
    def get(alg):
        key = '_'.join([k[1] for k in alg])

        return {
            'gol-cpu-naive_char': 'CPU Naive (char)',
            'gol-cpu-naive_int': 'CPU Naive (int)',
            'gol-cpu-bitwise-cols-naive-32': 'CPU Bitwise Cols Naive 32',
            'gol-cpu-bitwise-cols-naive-64': 'CPU Bitwise Cols Naive 64',
            'gol-cpu-bitwise-cols-macro-32': 'CPU Bitwise Cols Macro 32',
            'gol-cpu-bitwise-cols-macro-64': 'CPU Bitwise Cols Macro 64',
            'gol-cpu-bitwise-tiles-naive-32': 'CPU Bitwise Tiles Naive 32',
            'gol-cpu-bitwise-tiles-naive-64': 'CPU Bitwise Tiles Naive 64',
            'gol-cpu-bitwise-tiles-macro-32': 'CPU Bitwise Tiles Macro 32',
            'gol-cpu-bitwise-tiles-macro-64': 'CPU Bitwise Tiles Macro 64',

            'gol-cuda-naive_char': 'CUDA Naive (char)',
            'gol-cuda-naive_int': 'CUDA Naive (int)',
            'gol-cuda-naive-bitwise-no-macro-32': 'CUDA Bitwise Naive 32',
            'gol-cuda-naive-bitwise-no-macro-64': 'CUDA Bitwise Naive 64',
            'gol-cuda-naive-bitwise-cols-32': 'CUDA Bitwise Cols Macro 32',
            'gol-cuda-naive-bitwise-cols-64': 'CUDA Bitwise Cols Macro 64',
            'gol-cuda-naive-bitwise-tiles-32': 'CUDA Bitwise Tiles Macro 32',
            'gol-cuda-naive-bitwise-tiles-64': 'CUDA Bitwise Tiles Macro 64',
            'gol-cuda-local-one-cell-cols-32': 'CUDA Local Cols 32',
            'gol-cuda-local-one-cell-cols-64': 'CUDA Local Cols 64',
            'gol-cuda-local-one-cell-32--bit-tiles': 'CUDA Local Bit Tiles 32',
            'gol-cuda-local-one-cell-64--bit-tiles': 'CUDA Local Bit Tiles 64',

            'eff-baseline': 'Eff Baseline',
            'eff-baseline-shm': 'Eff Baseline SHM',
            'eff-sota-packed-32': 'Eff SOTA Packed 32',
            'eff-sota-packed-64': 'Eff SOTA Packed 64',

            'gol-cuda-local-one-cell-64--bit-tiles_no-work': 'CUDA Local Bit Tiles 64 No Work',
            'gol-cuda-local-one-cell-64--bit-tiles_full-work': 'CUDA Local Bit Tiles 64 Full Work',
            'gol-cuda-local-one-cell-64--bit-tiles_glider-gun': 'CUDA Local Bit Tiles 64 Glider Gun',
            'gol-cuda-local-one-cell-64--bit-tiles_spacefiller': 'CUDA Local Bit Tiles 64 Spacefiller',

        }[key]

class ALG_LIST:
    cpu_naive_char =                     [(ra.Key.algorithm_name, 'gol-cpu-naive'), (ra.Key.base_grid_encoding, 'char')]
    cpu_naive_int =                      [(ra.Key.algorithm_name, 'gol-cpu-naive'), (ra.Key.base_grid_encoding, 'int')]
    
    cpu_bitwise_cols_naive_32 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-naive-32')]
    cpu_bitwise_cols_naive_64 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-naive-64')]
    
    cpu_bitwise_cols_macro_32 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-macro-32')]
    cpu_bitwise_cols_macro_64 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-macro-64')]

    cpu_bitwise_tiles_naive_32 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-32')]    
    cpu_bitwise_tiles_naive_64 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-64')]

    cpu_bitwise_tiles_macro_32 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-32')]
    cpu_bitwise_tiles_macro_64 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-64')]
    
    cuda_naive_char =                    [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'char')]
    cuda_naive_int =                     [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'int')]

    cuda_naive_bitwise_no_macro_32 =     [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-32')]
    cuda_naive_bitwise_no_macro_64 =     [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-64')]
    
    cuda_naive_bitwise_cols_32 =         [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-32')]
    cuda_naive_bitwise_cols_64 =         [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-64')]

    cuda_naive_bitwise_tiles_32 =        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-tiles-32')]
    cuda_naive_bitwise_tiles_64 =        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-tiles-64')]
    
    cuda_local_one_cell_cols_32 =        [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-32')]
    cuda_local_one_cell_cols_64 =        [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-64')]

    cuda_local_one_cell_bit_tiles_32 =   [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-32--bit-tiles')]
    cuda_local_one_cell_bit_tiles_64 =   [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-64--bit-tiles')]

    eff_baseline = [(ra.Key.algorithm_name, 'eff-baseline')]
    eff_baseline_shm = [(ra.Key.algorithm_name, 'eff-baseline-shm')]
    
    eff_sota_packed_32 = [(ra.Key.algorithm_name, 'eff-sota-packed-32')]
    eff_sota_packed_64 = [(ra.Key.algorithm_name, 'eff-sota-packed-64')]

    ALGS = [
        cpu_naive_char,
        cpu_naive_int,
        cpu_bitwise_cols_naive_32,
        cpu_bitwise_cols_naive_64,
        cpu_bitwise_cols_macro_32,
        cpu_bitwise_cols_macro_64,
        cuda_naive_char,
        cuda_naive_int,
        cuda_naive_bitwise_no_macro_32,
        cuda_naive_bitwise_no_macro_64,
        cuda_naive_bitwise_cols_32,
        cuda_naive_bitwise_cols_64,
    ]

    g_1024 = [(ra.Key.grid_dimensions, '1024x1024')]
    g_2048 = [(ra.Key.grid_dimensions, '2048x2048')]
    g_4096 = [(ra.Key.grid_dimensions, '4096x4096')]
    g_8192 = [(ra.Key.grid_dimensions, '8192x8192')]
    g_16384 = [(ra.Key.grid_dimensions, '16384x16384')]
    g_32768 = [(ra.Key.grid_dimensions, '32768x32768')]
    g_65536 = [(ra.Key.grid_dimensions, '65536x65536')]

    data__no_work = [(ra.Key.tag, 'no-work')]
    data__full_work = [(ra.Key.tag, 'full-work')]
    data__glider_gun = [(ra.Key.tag, 'glider-gun')]
    data__spacefiller = [(ra.Key.tag, 'spacefiller')]

def from_ms_to_micro_seconds(val):
    return val * 1_000

class TimePerCellPerIter__InputSize:
    PLOT_NAME = 'time_per_cell_per_iter__input_size'

    def __init__(self, results: ra.Results):
        self.results = results
        self.tested_grids = []
        self.algs = []

    def set_algs(self, algs):
        self.algs = algs
        return self

    def set_grids(self, grids):
        self.tested_grids = grids
        return self

    def gen_graphs(self):
        plt.figure(figsize=(8, 6))

        means = []
        box_data = []
        
        for alg in self.algs:
            alg_values = []
            alg_box_data = []

            for grid in self.tested_grids:
                exp = self.results.get_experiments_with([*alg, *grid])

                if not exp:
                    alg_values.append(None)
                    alg_box_data.append(None)
                    continue

                if len(exp) != 1:
                    print('Found more than 1 exp:', len(exp), 'for:', alg, grid)

                measurements = exp[0].get_measurement_set()
                
                val = measurements.get_median(lambda m: m.compute_runtime_per_cell_per_iter())

                if val is None:
                    print ('None value for:', alg, grid)
                    val = 0

                time_per_million = val * 1_000_000

                all_vals = measurements.get_valid_vals(lambda m: m.compute_runtime_per_cell_per_iter())
                alg_values.append(time_per_million)
                
                b_data = [v * 1_000_000 for v in all_vals]
                
                # for i in range(len(b_data)):
                #     random_coef = random.uniform(0.5, 1.5)
                #     b_data[i] *= random_coef

                alg_box_data.append(b_data)

            means.append(alg_values)
            box_data.append(alg_box_data)

        x_labels = [grid[0][1] for grid in self.tested_grids]
        x_positions = range(len(x_labels))
        
        for i, (mean_vals, dist_vals) in enumerate(zip(means, box_data)):
            mean_vals = [from_ms_to_micro_seconds(v) for v in mean_vals]
            plt.plot(x_positions, mean_vals, label=str(self.algs[i]))

        # for i, (mean_vals, dist_vals) in enumerate(zip(means, box_data)):
        #     for j, mean_val in enumerate(mean_vals):
        #         if mean_val is not None and dist_vals[j] is not None:
        #             plt.boxplot(dist_vals[j], positions=[j], widths=0.07)

        plt.xticks(x_positions, x_labels, rotation=45)

        plt.xlabel("Grid Size")
        plt.ylabel("Time / Million Cells / One Iteration (µs)")
        plt.ylim(bottom=0)
        plt.legend([LegendNames.get(alg) for alg in self.algs])

        out_path = os.path.join(GRAPH_DIR, PLOT_NAME_template.format(plot_type=self.PLOT_NAME))

        plt.tight_layout()
        plt.savefig(out_path, format=MODE)


class CompareAlgsOnGrids:
    PLOT_NAME = 'compare_algs_on_data'

    def __init__(self, results: ra.Results):
        self.results = results
        self.tested_grids = []
        self.algs = []
        self.base_algs = []
        self.data_loaders = []
        self.x_labels = []

    def set_base_algs(self, algs):
        self.base_algs = algs
        return self

    def set_algs(self, algs):
        self.algs = algs
        return self

    def set_grid(self, grid):
        self.tested_grid = grid
        return self
    
    def set_data_loaders(self, data_loaders):
        self.data_loaders = data_loaders
        return self

    def set_x_labels(self, x_labels):
        self.x_labels = x_labels
        return self

    def gen_graphs(self):
        plt.figure(figsize=(8, 6))

        x = np.arange(len(self.data_loaders)) * 1.2
        bar_width = 0.5 / len(self.algs)

        for i, alg in enumerate(self.base_algs + self.algs):
            y_vals = []
            for loader in self.data_loaders:
                is_base = i < len(self.base_algs)
                
                if is_base:
                    grid = [*self.tested_grid]
                else:
                    grid = [*self.tested_grid, *loader]

                exp = self.results.get_experiments_with([*alg, *grid])
                if not exp:
                    y_vals.append(0)
                    continue

                measurements = exp[0].get_measurement_set()
                val = measurements.get_median(lambda m: m.compute_runtime_per_cell_per_iter())
                if val is None:
                    val = 0

                y_vals.append(val * 1_000_000)

            y_vals = [from_ms_to_micro_seconds(v) for v in y_vals]
            plt.bar(x + i * bar_width - bar_width / 2, y_vals, bar_width, label=LegendNames.get(alg))
            

        plt.xticks(x + bar_width * (len(self.algs) / 2), self.x_labels, rotation=45)
        plt.xlabel("Data on grid " + self.tested_grid[0][1])
        plt.ylabel("Time / Million Cells / One Iteration (µs)")
        plt.legend()
        out_path = os.path.join(GRAPH_DIR, PLOT_NAME_template.format(plot_type=self.PLOT_NAME))
        plt.tight_layout()
        plt.savefig(out_path, format=MODE)


def combined(alg, data):
    return [*alg, *data]

results = ra.Results.from_directory(BASE_DIR)

TimePerCellPerIter__InputSize(results) \
    .set_algs([
        
        # ALG_LIST.cpu_naive_char,
        # ALG_LIST.cpu_naive_int,
        # ALG_LIST.cpu_bitwise_cols_naive_32,
        # ALG_LIST.cpu_bitwise_cols_naive_64,
        # ALG_LIST.cpu_bitwise_cols_macro_32,
        # ALG_LIST.cpu_bitwise_cols_macro_64,
        # ALG_LIST.cpu_bitwise_tiles_naive_32,
        # ALG_LIST.cpu_bitwise_tiles_naive_64,
        # ALG_LIST.cpu_bitwise_cols_macro_32,
        # ALG_LIST.cpu_bitwise_cols_macro_64,
        

        # ALG_LIST.cuda_naive_char,
        # ALG_LIST.cuda_naive_int,
        
        # ALG_LIST.cuda_naive_bitwise_no_macro_32,
        # ALG_LIST.cuda_naive_bitwise_no_macro_64,

        # ALG_LIST.cuda_naive_bitwise_cols_32,
        # ALG_LIST.cuda_naive_bitwise_cols_64,

        # ALG_LIST.cuda_naive_bitwise_tiles_32,
        ALG_LIST.cuda_naive_bitwise_tiles_64,
        
        # ALG_LIST.cuda_local_one_cell_cols_32,
        # ALG_LIST.cuda_local_one_cell_cols_64,
        # ALG_LIST.cuda_local_one_cell_bit_tiles_32,
        # ALG_LIST.cuda_local_one_cell_bit_tiles_64,

        # ALG_LIST.eff_baseline,
        # ALG_LIST.eff_baseline_shm,
        # ALG_LIST.eff_sota_packed_32,
        # ALG_LIST.eff_sota_packed_64,
        
        combined(ALG_LIST.cuda_local_one_cell_bit_tiles_64, ALG_LIST.data__no_work),
        combined(ALG_LIST.cuda_local_one_cell_bit_tiles_64, ALG_LIST.data__full_work),
        combined(ALG_LIST.cuda_local_one_cell_bit_tiles_64, ALG_LIST.data__glider_gun),
        combined(ALG_LIST.cuda_local_one_cell_bit_tiles_64, ALG_LIST.data__spacefiller),

        # *ALG_LIST.ALGS
    ]) \
    .set_grids([
        # ALG_LIST.g_1024,
        # ALG_LIST.g_2048,
        ALG_LIST.g_4096,
        ALG_LIST.g_8192,
        ALG_LIST.g_16384,
        ALG_LIST.g_32768,
        ALG_LIST.g_65536,
    ]) \
    .gen_graphs()

CompareAlgsOnGrids(results) \
    .set_base_algs([
        ALG_LIST.cuda_naive_bitwise_cols_32,
        ALG_LIST.cuda_naive_bitwise_cols_64,

        ALG_LIST.cuda_naive_bitwise_tiles_32,
        ALG_LIST.cuda_naive_bitwise_tiles_64,
    ]) \
    .set_algs([

        ALG_LIST.cuda_local_one_cell_cols_32,
        ALG_LIST.cuda_local_one_cell_cols_64,

        ALG_LIST.cuda_local_one_cell_bit_tiles_32,
        ALG_LIST.cuda_local_one_cell_bit_tiles_64,

    ]) \
    .set_data_loaders([
        ALG_LIST.data__full_work,
        ALG_LIST.data__spacefiller,
        ALG_LIST.data__glider_gun,
        ALG_LIST.data__no_work,
    ]) \
    .set_x_labels([
        'Full Work',
        'Spacefiller',
        'Glider Gun',
        'No Work',
    ]) \
    .set_grid(
        # ALG_LIST.g_1024,
        # ALG_LIST.g_2048,
        # ALG_LIST.g_4096,
        ALG_LIST.g_8192,
        # ALG_LIST.g_16384,
        # ALG_LIST.g_32768,
        # ALG_LIST.g_65536,
    ) \
    .gen_graphs()
