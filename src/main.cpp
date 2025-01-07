#include "algorithms/_shared/bitwise-cols/bit_cols_grid.hpp"
#include "algorithms/_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include "algorithms/cpu-bitwise-cols-macro/gol_cpu_bitwise_cols_macro.hpp"
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

void print_fail(const std::string& name, std::uint16_t expected, std::uint16_t result) {
    std::cout << "\033[1;31m" << name << "\033[0m" << std::endl;
    std::cout << "Expected: " << std::bitset<16>(expected) << std::endl;
    std::cout << "Result:   " << std::bitset<16>(result) << std::endl;
}

void print_pass(const std::string& name) {
    std::cout << "\033[1;32m" << name << "\033[0m" << std::endl;
}

void test_inner() {

    std::string test_name = "";
    std::uint16_t lt = 0, ct = 0, rt = 0;
    std::uint16_t lc = 0, cc = 0, rc = 0;
    std::uint16_t lb = 0, cb = 0, rb = 0;

    std::uint16_t ex = 0;

    // ================================

    test_name = "Test - make alive (1)";

    lc = 0b0000'1000'0000;
    cc = 0b0010'0000'0000;
    rc = 0b0001'0000'0000;

    ex = 0b0001'0000'0000;

    std::uint16_t res =
        algorithms::MacroBitOperations<std::uint16_t>::compute_center_col(lc, cc, rc, lc, cc, rc, lc, cc, rc);

    if (res != ex) {
        print_fail(test_name, ex, res);
    }
    else {
        print_pass(test_name);
    }

    // ================================

    test_name = "Test - make alive (2)";

    lc = 0b0000'0000'0001;
    cc = 0b0000'0000'0100;
    rc = 0b0000'0000'0010;

    ex = 0b0000'0000'0100;
}

int main() {
    std::cout << "Hello" << std::endl;

    // test_inner(); return 0;

    infrastructure::ExperimentParams params = {
        // .algorithm_name = "gol-cpu-naive",
        .algorithm_name = "gol-cpu-bitwise-cols-64",
        // .algorithm_name = "gol-cpu-bitwise-cols-macro-64",
        // .algorithm_name = "gol-cuda-naive",
        // .grid_dimensions = {10'000, 10'000},
        .grid_dimensions = {512, 1024},
        // .grid_dimensions = {8, 32},
        .iterations = 10,
        .data_loader_name = "random-ones-zeros",
        // .data_loader_name = "one-glider-in-the-conner",
        // .debug_logs = true,

        .measure_speedup = true,
        // .speedup_bench_algorithm_name = "gol-cpu-naive",
        .speedup_bench_algorithm_name = "gol-cpu-bitwise-cols-64",

        .validate = true,
        .print_validation_diff = true,
    };

    infrastructure::ExperimentManager manager;

    manager.run(params);

    return 0;
}
