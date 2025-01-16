import results_abstraction as ra

HP_DIR = './hyper-params-measurements/results'
# HP_DIR = './experiments-outputs'

results = ra.Results.from_directory(HP_DIR)

class CudaNaiveCases:
    cuda_naive_char = [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'char')]
    cuda_naive_int = [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'int')]

    cuda_naive_bitwise_32 = [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-32')]
    cuda_naive_bitwise_64 = [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-64')]

    cuda_bitwise_macro_32 = [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-32')]
    cuda_bitwise_macro_64 = [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-64')]

    algs = [
        cuda_naive_char,
        cuda_naive_int,
        cuda_naive_bitwise_32,
        cuda_naive_bitwise_64,
        cuda_bitwise_macro_32,
        cuda_bitwise_macro_64,
    ]

    tested_grids = [
        (ra.Key.grid_dimensions, '8192x8192'),
        (ra.Key.grid_dimensions, '16384x16384'),
        (ra.Key.grid_dimensions, '32768x32768'),
        (ra.Key.grid_dimensions, '65536x65536'),
    ]

    @staticmethod
    def get_best_block_sizes():
        bests = []

        for alg in CudaNaiveCases.algs:
            for grid in CudaNaiveCases.tested_grids:

                best_exp = CudaNaiveCases._get_best_experiment_for(alg, grid)

                if best_exp is None:
                    continue

                thread_block_size = best_exp.get_param(ra.Key.thread_block_size)
                perf_per_iter = best_exp.get_median_runtime_per_iter()

                bests.append((alg, grid, thread_block_size, perf_per_iter))

        return bests        
                
    @staticmethod
    def _get_best_experiment_for(alg, grid) -> ra.Experiment:
        exps = results.get_experiments_with([*alg, grid])

        if len(exps) == 0:
            return None

        best_exp = None

        for exp in exps:
            median_perf = exp.get_median_runtime_per_iter()
            
            if median_perf is None:
                continue

            if best_exp is None:
                best_exp = exp
                continue
            
            if exp.get_median_runtime_per_iter() < median_perf:
                best_exp = exp

        return best_exp
    
    @staticmethod
    def print_stats_for_all():
        for alg in CudaNaiveCases.algs:
            print('Algorithm:', alg, '\n')

            for grid in CudaNaiveCases.tested_grids:
                print('  Grid:', grid[1])

                CudaNaiveCases._print_stats_for(alg, grid)

    @staticmethod
    def _print_stats_for(alg, grid):
        exps = results.get_experiments_with([*alg, grid])
        exps = [exp for exp in exps if exp.get_median_runtime_per_iter() is not None]

        exps.sort(key=lambda e: e.get_median_runtime_per_iter())

        for exp in exps:
            print(f'     {exp.get_param(ra.Key.thread_block_size)}: {exp.get_median_runtime_per_iter()}')

        print()

                

# bests = CudaNaiveCases.get_best_block_sizes()

# for best in bests:
#     print(best)

CudaNaiveCases.print_stats_for_all()
