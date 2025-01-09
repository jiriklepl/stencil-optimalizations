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

    lexicon::Lexicon lexicon;
    
    infrastructure::Grid<2, char> grid(20, 20);
    lexicon.insert_patters(grid, "glider[1,1] glider[0,7]");

    std::cout << pretty(grid) << std::endl;

    return 0;

    // infrastructure::ExperimentParams params = {
    //     //////////////////////////////
    //     // TESTED ALGORITHM         //
    //     //////////////////////////////

    //     // .algorithm_name = "gol-cpu-naive",
    //     // .algorithm_name = "gol-cpu-bitwise-cols-64",
    //     // .algorithm_name = "gol-cpu-bitwise-cols-macro-64",
    //     // .algorithm_name = "gol-cuda-naive-bitwise-cols-64",
    //     // .algorithm_name = "an5d-cpu-64",
    //     .algorithm_name = "an5d-cuda-64",
    //     // .algorithm_name = "gol-cuda-naive",
        
    //     //////////////////////////////
    //     // SPACE                    //
    //     //////////////////////////////
        
    //     // .grid_dimensions = {10'000, 10'000},
    //     .grid_dimensions = {512 * (64 + 32), 1024 * (64 + 32)},
    //     // .grid_dimensions = {64, 128},
    //     // .grid_dimensions = {64, 256},
    //     .iterations = 100'000,

    //     //////////////////////////////
    //     // DATA                     //
    //     //////////////////////////////

    //     .data_loader_name = "random-ones-zeros",
    //     // .data_loader_name = "one-glider-in-the-conner",

    //     //////////////////////////////
    //     // SPEEDUP                  //
    //     //////////////////////////////

    //     .measure_speedup = true,
    //     // .speedup_bench_algorithm_name = "gol-cpu-naive",
    //     // .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-64",
    //     // .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-macro-64",
    //     .speedup_bench_algorithm_name = "gol-cuda-naive-bitwise-cols-64",

    //     //////////////////////////////
    //     // VALIDATION               //
    //     //////////////////////////////

    //     .validate = true,
    //     // .print_validation_diff = true,
    //     .validation_algorithm_name = "gol-cuda-naive",
    // };

    // infrastructure::ExperimentManager manager;

    // manager.run(params);

    // return 0;
}
