#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-tiles/bitwise_tiles_gol_operations_tests.hpp"
#include "infrastructure/data_loader.hpp"
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

    // tests::BitwiseTileOpsTests::run();

    infrastructure::ExperimentParams params = {
        .algorithm_name = "gol-cpu-naive",
        .grid_dimensions = {10, 128},
        .iterations = 50,
    };

    infrastructure::OneGliderInTheConnerLoader loader;
    auto grid = loader.load_data<2, char>(params);
    algorithms::BitColsGrid bit_cols_grid(grid);

    std::cout << bit_cols_grid.debug_print() << std::endl;

    std::cout << "--------" << std::endl;

    std::cout << pretty(grid) << std::endl;

    std::cout << "--------" << std::endl;

    std::cout << pretty(bit_cols_grid.bit_cols_grid) << std::endl;

    // infrastructure::ExperimentManager manager;

    // manager.run(params);

    return 0;
}
