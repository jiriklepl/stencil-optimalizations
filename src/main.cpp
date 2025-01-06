#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include "algorithms/cpu-bitwise-cols/gol_cpu_bitwise_cols.hpp"
#include "infrastructure/data_loader.hpp"
#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/grid.hpp"
#include <bit>
#include <bitset>
#include <cstdint>

#include <iostream>

#include "debug_utils/pretty_print.hpp"
using namespace debug_utils;

int main() {
    std::cout << "Hello" << std::endl;

    // tests::BitwiseTileOpsTests::run();

    infrastructure::ExperimentParams params = {
        // .algorithm_name = "gol-cpu-naive",
        .algorithm_name = "gol-cpu-bitwise-cols-16",
        .grid_dimensions = {30, 64},
        .iterations = 1,
        .data_loader_name = "random-ones-zeros",
        // .data_loader_name = "one-glider-in-the-conner",
        // .debug_logs = true,

        .validate = true,
        .print_validation_diff = true,
    };

    infrastructure::ExperimentManager manager;

    manager.run(params);

    return 0;
}
