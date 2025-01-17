import results_abstraction as ra

HP_DIR = './hyper-params-measurements/ampere'
# HP_DIR = './experiments-outputs'

class Algs:

    # NAIVE

    GPU_naive = [
        [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'char')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'int')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-32')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-64')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-32')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-64')],
    ]

    GPU_naive_TEST_GRIDS = [
        [(ra.Key.grid_dimensions, '8192x8192')],
        [(ra.Key.grid_dimensions, '16384x16384')],
        [(ra.Key.grid_dimensions, '32768x32768')],
        [(ra.Key.grid_dimensions, '65536x65536')],
    ]

    GPU_naive_hyper_params = [
        ra.Key.thread_block_size,
    ]

    # LOCAL

    GPU_local = [
        [(ra.Key.algorithm_name, 'gol-cuda-naive-local-32')],
        [(ra.Key.algorithm_name, 'gol-cuda-naive-local-64')],
    ]

    GPU_local_TEST_GRIDS = [
        [(ra.Key.grid_dimensions, '32768x32768'), (ra.Key.data_loader_name, 'lexicon')],
        [(ra.Key.grid_dimensions, '32768x32768'), (ra.Key.data_loader_name, 'zeros')],
        [(ra.Key.grid_dimensions, '32768x32768'), (ra.Key.data_loader_name, 'always-changing')],

        [(ra.Key.grid_dimensions, '65536x65536'), (ra.Key.data_loader_name, 'lexicon')],
        [(ra.Key.grid_dimensions, '65536x65536'), (ra.Key.data_loader_name, 'zeros')],
        [(ra.Key.grid_dimensions, '65536x65536'), (ra.Key.data_loader_name, 'always-0changing')],
    ]

    GPU_local_hyper_params = [
        ra.Key.thread_block_size,

        ra.Key.warp_dims_x,
        ra.Key.warp_dims_y,
        
        ra.Key.warp_tile_dims_x,
        ra.Key.warp_tile_dims_y,
        
        ra.Key.streaming_direction,
        
        ra.Key.state_bits_count,
    ]    

class BestHP:
    def __init__(self, results: ra.Results, algs: list[list[tuple[str, str]]], tested_grids: list[list[tuple[str, str]]]):
        self.results: ra.Results = results
        self.algs = algs
        self.tested_grids = tested_grids

        self.print_limit_per_test_case = 400

    def print_best(self, hyper_params_keys: list[str]):
        for alg in self.algs:
            print('Algorithm:', alg, '\n')

            for grid in self.tested_grids:
                print('  Grid:', grid)

                bests = self._get_best_experiments_for(alg, grid)
                self._print_bests(bests, hyper_params_keys)

                print()

    def _get_best_experiments_for(self, alg, grid) -> ra.Experiment:
        exps = self.results.get_experiments_with([*alg, *grid])
        exps = [exp for exp in exps if exp.get_median_runtime_per_iter() is not None]

        exps.sort(key=lambda e: e.get_median_runtime_per_iter())
    
        return exps
    

    def _print_bests(self, exps: list[ra.Experiment], hyper_params_keys: list[str]):
        for exp in exps[:self.print_limit_per_test_case]:
            print(f'    rt: {exp.get_median_runtime_per_iter()}', end=' ')

            for key in hyper_params_keys:
                print(f'{key}: {exp.get_param(key)}\t', end=' ')

            print()


results = ra.Results.from_directory(HP_DIR)

best_hp_naive_cuda = BestHP(results, Algs.GPU_naive, Algs.GPU_naive_TEST_GRIDS)
best_hp_local_cuda = BestHP(results, Algs.GPU_local, Algs.GPU_local_TEST_GRIDS)

# best_hp_naive_cuda.print_best(Algs.GPU_naive_hyper_params)
best_hp_local_cuda.print_best(Algs.GPU_local_hyper_params)