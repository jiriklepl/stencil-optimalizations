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

template <std::size_t N>
void test_f() {
    std::cout << "N: " << N << std::endl;
}

template <std::size_t I, std::size_t N>
struct static_for {
    template <typename Functor>
    static void run(Functor&& f) {
        f.template operator()<I>();
        static_for<I + 1, N>::run(std::forward<Functor>(f));
    }
};

template <std::size_t N>
struct static_for<N, N> {
    template <typename Functor>
    static void run(Functor&&) {
    }
};

int main() {
    std::cout << "Hello" << std::endl;

    // tests::BitwiseTileOpsTests::run();

    infrastructure::ExperimentParams params = {
        .algorithm_name = "gol-cpu-naive",
        .grid_dimensions = {30, 64},
        .iterations = 65,
    };

    infrastructure::OneGliderInTheConnerLoader loader;
    auto grid = loader.load_data<2, char>(params);

    algorithms::GoLCpuBitwiseCols<16> algo;

    algo.set_and_format_input_data(grid);
    algo.initialize_data_structures();
    algo.run(params.iterations);
    algo.finalize_data_structures();

    auto res = algo.fetch_result();

    // std::cout << pretty(res) << std::endl;

    // std::cout << bit_cols_grid.debug_print() << std::endl;

    // std::cout << "--------" << std::endl;

    // std::cout << pretty(grid) << std::endl;

    // std::cout << "--------" << std::endl;

    // infrastructure::ExperimentManager manager;

    // manager.run(params);

    return 0;
}
