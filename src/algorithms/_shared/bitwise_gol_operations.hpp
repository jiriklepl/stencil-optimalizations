#ifndef ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP
#define ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include "../_shared/gol_bit_grid.hpp"
#include <bitset>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <thread>

namespace algorithms {

struct BitwiseOps {
    using tile_type = std::uint64_t;

    template <tile_type... rows>
    static constexpr tile_type tile_num() {
        tile_type result = 0;
        ((result = (result << 8) | rows), ...);
        return result;
    }

    // clang-format off
    static tile_type compute_center_tile(
        tile_type lt, tile_type ct, tile_type rt, 
        tile_type lc, tile_type cc, tile_type rc,
        tile_type lb, tile_type cb, tile_type rb) {
        
        tile_type result = 0;

        auto neighborhood_mask = tile_num<
            0b0000'0111,
            0b0000'0101,
            0b0000'0111>();

        auto cell_mask = tile_num<
            0b0000'0000,
            0b0000'0010,
            0b0000'0000>();

        auto cell_setter = tile_num<
            0b0000'0000,
            0b0000'0010,
            0b0000'0000>();

        auto cell_zeroer = tile_num<
            0b1111'1111,
            0b1111'1111,
            0b1111'1111,
            0b1111'1111,
            0b1111'1111,
            0b1111'1111,
            0b1111'1101,
            0b1111'1111>();

        auto current = cc;

        for (std::size_t row = 0; row < 7; ++row) {
            for (std::size_t col = 0; col < 7; ++col) {

                auto neighborhood = current & neighborhood_mask;
                auto cell = current & cell_mask;

                auto alive_neighbours = __builtin_popcount(neighborhood);

                if (cell != 0) {
                    if (alive_neighbours < 2 || alive_neighbours > 3) {
                        result &= cell_zeroer;
                    }
                    else {
                        result |= cell_setter;
                    }
                }
                else {
                    if (alive_neighbours == 3) {
                        result |= cell_setter;
                    }
                    else {
                        result &= cell_zeroer;
                    }
                }

                cell_setter <<= 1;
                cell_zeroer = (cell_zeroer << 1) | 0b1;
                current >>= 1;

                // std::cout << "row: " << 6 - row << ", col: " << 6 - col << std::endl;
                // std::cout << debug_print(result) << std::endl;
            }

            cell_setter <<= 2;
            cell_zeroer = (cell_zeroer << 2) | 0b11;
            current >>= 2;
        }
        std::cout << std::endl;
        return result;
    }
    // clang-format on

    static std::string debug_print(tile_type tile) {
        std::ostringstream oss;

        for (int i = 7; i >= 0; --i) {
            std::ostringstream line;
            line << std::bitset<8>(tile >> (8 * i));

            for (char ch : line.str()) {
                if (ch == '0') {
                    oss << "\033[30m" << ch << "\033[0m";
                }
                else {
                    oss << "\033[31m" << ch << "\033[0m";
                }
            }

            oss << std::endl;
        }

        return oss.str();
    }
};
} // namespace algorithms

#endif // ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP