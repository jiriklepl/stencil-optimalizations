#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include "algorithms/cpu-bitwise-cols-macro/gol_cpu_bitwise_cols_macro.hpp"
#include "algorithms/cpu-bitwise-cols/gol_cpu_bitwise_cols.hpp"
#include "algorithms/cuda-naive-bitwise/gol_cuda_naive_bitwise.hpp"
#include "infrastructure/data_loader.hpp"
#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/grid.hpp"
#include <bit>
#include <bitset>
#include <cstdint>

#include "debug_utils/pretty_print.hpp"
#include <iostream>

using namespace debug_utils;

int main() {
    std::cout << "Hello" << std::endl;

    infrastructure::ExperimentParams params = {
        // .algorithm_name = "gol-cpu-naive",
        // .algorithm_name = "gol-cpu-bitwise-cols-64",
        // .algorithm_name = "gol-cpu-bitwise-cols-macro-32",
        .algorithm_name = "gol-cuda-naive-bitwise-cols-64",
        // .algorithm_name = "gol-cuda-naive",
        // .grid_dimensions = {10'000, 10'000},
        .grid_dimensions = {512, 1024},
        // .grid_dimensions = {8, 32},
        .iterations = 100,
        .data_loader_name = "random-ones-zeros",
        // .data_loader_name = "one-glider-in-the-conner",
        // .debug_logs = true,

        .measure_speedup = true,
        // .speedup_bench_algorithm_name = "gol-cpu-naive",
        .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-32",

        .validate = true,
        .print_validation_diff = true,
    };

    infrastructure::ExperimentManager manager;

    manager.run(params);

    return 0;
}
