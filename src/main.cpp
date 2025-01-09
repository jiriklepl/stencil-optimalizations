#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include "algorithms/cpu-bitwise-cols-macro/gol_cpu_bitwise_cols_macro.hpp"
#include "algorithms/cpu-bitwise-cols/gol_cpu_bitwise_cols.hpp"
#include "algorithms/cuda-naive-bitwise/gol_cuda_naive_bitwise.hpp"
#include "infrastructure/data_loader.hpp"
#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/gol-lexicon/lexicon.hpp"
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
        //////////////////////////////
        // TESTED ALGORITHM         //
        //////////////////////////////

        .algorithm_name = "gol-cpu-naive",
        // .algorithm_name = "gol-cpu-bitwise-cols-64",
        // .algorithm_name = "gol-cpu-bitwise-cols-macro-64",
        // .algorithm_name = "gol-cuda-naive-bitwise-cols-64",
        // .algorithm_name = "an5d-cpu-64",
        // .algorithm_name = "an5d-cuda-64",
        // .algorithm_name = "gol-cuda-naive",

        //////////////////////////////
        // SPACE                    //
        //////////////////////////////

        // .grid_dimensions = {10'000, 10'000},
        // .grid_dimensions = {512 * (64 + 32), 1024 * (64 + 32)},
        // .grid_dimensions = {512, 1024},
        // .grid_dimensions = {64, 128},
        // .grid_dimensions = {64, 256},
        .grid_dimensions = {32, 16},

        .iterations = 10,

        //////////////////////////////
        // DATA                     //
        //////////////////////////////

        // .data_loader_name = "random-ones-zeros",
        .data_loader_name = "lexicon",

        .pattern_expression = "glider[0,1]",

        //////////////////////////////
        // SPEEDUP                  //
        //////////////////////////////

        // .measure_speedup = true,
        // .speedup_bench_algorithm_name = "gol-cpu-naive",
        // .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-64",
        // .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-macro-64",
        .speedup_bench_algorithm_name = "gol-cuda-naive-bitwise-cols-64",

        //////////////////////////////
        // VALIDATION               //
        //////////////////////////////

        // .validate = true,
        // .print_validation_diff = true,
        .validation_algorithm_name = "gol-cuda-naive",

        .animate_output = true,
    };

    infrastructure::ExperimentManager manager;

    manager.run(params);

    return 0;
}
