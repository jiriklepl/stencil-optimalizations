#include "algorithms/_shared/bitwise_gol_operations_tests.hpp"
#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/grid.hpp"
#include <bit>
#include <bitset>
#define DEBUG

#include <iostream>

#include "debug_utils/pretty_print.hpp"
using namespace debug_utils;

using tile_type = uint64_t;

template <tile_type... rows>
constexpr tile_type tile_num() {
    tile_type result = 0;
    ((result = (result << 8) | rows), ...);
    return result;
}

template <tile_type CONST>
void print_const() {
    std::cout << std::bitset<64>(CONST) << std::endl;
}

int main() {
    std::cout << "Hello" << std::endl;

    tests::BitwiseOpsTests::run();

    infrastructure::ExperimentParams params = {
        .algorithm_name = "gol-cpu-naive",
        .grid_dimensions = {10, 20},
        .iterations = 50,
    };

    // infrastructure::ExperimentManager manager;

    // manager.run(params);

    return 0;
}
