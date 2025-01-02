#ifndef BITWISE_GOL_OPERATIONS_TESTS_HPP
#define BITWISE_GOL_OPERATIONS_TESTS_HPP

#include "./bitwise_tiles_gol_operations.hpp"
#include <bits/c++config.h>
#include <cstdint>
#include <iostream>

namespace tests {

using namespace algorithms;

class BitwiseTileOpsTests {
  public:
    using tile_type = std::uint64_t;

    static void run() {
        perform("center of the tile is computed correctly", t_compute_center_tile);
    }

  private:
    template <typename TestFunc>
    static void perform(const std::string& name, TestFunc&& func) {
        std::cout << "\033[34mRunning test: \033[33m" << name << "\033[0m" << std::endl;
        auto passed = func();

        if (passed) {
            std::cout << "\033[32m>> Test passed\033[0m" << std::endl;
        }
        else {
            std::cout << "\033[31m>> Test failed\033[0m" << std::endl;
        }
    }

    // clang-format off
    static bool t_compute_center_tile() {        
        // arrange

        tile_type lt = 0, ct = 0, rt = 0;
        tile_type lc = 0, cc = 0, rc = 0;
        tile_type lb = 0, cb = 0, rb = 0;

        // game of live glider
        cc = BitwiseTileOps::tile_num<
            0b0000'0000,
            0b0010'0000,
            0b0001'0000,
            0b0111'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000>();

        // act
        auto actual = BitwiseTileOps::compute_center_tile(
            lt, ct, rt,
            lc, cc, rc,
            lb, cb, rb);

        // assert
        auto expected = BitwiseTileOps::tile_num<
            0b0000'0000,
            0b0000'0000,
            0b0101'0000,
            0b0011'0000,
            0b0010'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000>();

        return assert_with_print(actual, expected);
    }

    static bool t_compute_top_site() {
        // arrange
        
        tile_type lt = 0, ct = 0, rt = 0;
        tile_type lc = 0, cc = 0, rc = 0;
        tile_type lb = 0, cb = 0, rb = 0;

        return true;
    };

    static bool assert_with_print(tile_type actual, tile_type expected) {
        auto succeeded = actual == expected;

        if (!succeeded) {
            std::cout << "Expected: " << std::endl;
            std::cout << BitwiseTileOps::debug_print(expected) << std::endl;

            std::cout << "Actual: " << std::endl;
            std::cout << BitwiseTileOps::debug_print(actual) << std::endl;
        }

        return succeeded;
    }

    // clang-format on
};

} // namespace tests

#endif // BITWISE_GOL_OPERATIONS_TESTS_HPP