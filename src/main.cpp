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

    auto x = 0b0000'0111 << 3;

    std::cout << std::bitset<8>(x) << std::endl;

    auto num = tile_num<0b0000'0111, 0b0000'0101, 0b0000'0111>();

    auto expected = 0b0000'0111'0000'0101'0000'0111;
    std::cout << "Num:      " << std::bitset<24>(num) << std::endl;
    std::cout << "Expected: " << std::bitset<24>(expected) << std::endl;

    infrastructure::ExperimentParams params = {
        .algorithm_name = "gol-cpu-naive",
        .grid_dimensions = {10, 20},
        .iterations = 50,
    };

    // infrastructure::ExperimentManager manager;

    // manager.run(params);

    return 0;
}
