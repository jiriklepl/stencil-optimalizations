#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include "algorithms/_shared/common_grid_types.hpp"
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
#include <string>

using namespace debug_utils;


int main(int argc, char** argv) {
    std::cout << "Hello" << std::endl;

    auto params = infrastructure::ParamsParser::parse(argc, argv);

    c::set_colorful(params.colorful);

    std::cout << params.pretty_print() << std::endl;

    infrastructure::ExperimentManager<common::INT> manager;
    manager.run(params);

    return 0;
}
