#ifndef GOL_BIT_COL_GRID_HPP
#define GOL_BIT_COL_GRID_HPP

#include "../../../infrastructure/grid.hpp"
#include <bitset>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace algorithms {

class BitColsGrid {
  public:
    using bit_col_type = uint64_t;
    constexpr static std::size_t BITS_IN_COL = sizeof(bit_col_type) * 8;

    using size_type = std::size_t;
    using Grid = infrastructure::Grid<2, char>;

    BitColsGrid(const Grid& grid) {
        assert_dim_has_correct_size(grid);

        x_size = grid.size_in<0>();
        y_size = grid.size_in<1>() / BITS_IN_COL;

        bit_cols_grid.resize(x_size * y_size, 0);
        fill_grid(grid);
    }

  public:
    bit_col_type get_bit_col(std::size_t x, std::size_t y) const {
        return bit_cols_grid[idx(x, y)];
    }

    void set_bit_col(std::size_t x, std::size_t y, bit_col_type bit_col) {
        bit_cols_grid[idx(x, y)] = bit_col;
    }

    std::string debug_print() {
        std::string result;

        for (std::size_t y = 0; y < y_size; ++y) {
            for (std::size_t bit = 0; bit < BITS_IN_COL; ++bit) {
                for (std::size_t x = 0; x < x_size; ++x) {
                    auto col = get_bit_col(x, y);
                    char bit_char = ((col >> bit) & 1) ? '1' : '0';

                    result += color_0_1(bit_char);
                    result += " ";
                }
                result += "\n";
            }
            result += "\n";
        }

        return result;
    }

  private:
    void assert_dim_has_correct_size(const Grid& grid) {
        if (grid.size_in<1>() % 64 != 0) {
            throw std::invalid_argument("Grid dimensions must be a multiple of 8");
        }
    }

    void fill_grid(const Grid& grid) {
        for (std::size_t y = 0; y < grid.size_in<1>(); y += BITS_IN_COL) {
            for (std::size_t x = 0; x < grid.size_in<0>(); ++x) {
                bit_col_type bit_col = 0;

                for (std::size_t i = 0; i < BITS_IN_COL; ++i) {
                    auto value = static_cast<bit_col_type>(grid[x][y + i]);
                    bit_col |= value << i;

                    if (value != 0) {
                        std::cout << "x: " << x << " y: " << y << " i: " << i << " value: " << value << std::endl;
                    }
                }

                std::cout << "bits: " << std::bitset<64>(bit_col) << " at: " << x << " " << y << std::endl;

                set_bit_col(x, y, bit_col);
            }
        }
    }

    std::string color_0_1(char ch) {
        if (ch == '0') {
            return "\033[30m" + std::string(1, ch) + "\033[0m";
        }
        else {
            return "\033[31m" + std::string(1, ch) + "\033[0m";
        }
    }

    std::size_t idx(std::size_t x, std::size_t y) const {
        return y * x_size + x;
    }

    std::vector<bit_col_type> bit_cols_grid;
    size_type x_size;
    size_type y_size;
};

} // namespace algorithms

#endif // GOL_BIT_COL_GRID_HPP