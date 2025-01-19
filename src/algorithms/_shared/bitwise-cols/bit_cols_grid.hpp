#ifndef GOL_BIT_COL_GRID_HPP
#define GOL_BIT_COL_GRID_HPP

#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../../debug_utils/pretty_print.hpp"
#include "../../../infrastructure/grid.hpp"

using namespace debug_utils;

namespace algorithms {

template <typename bit_type>
struct BitColumns {
    constexpr static std::size_t X_BITS = 1;
    constexpr static std::size_t Y_BITS = sizeof(bit_type) * 8;

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        (void)x;

        bit_type one = 1;
        auto y_bit_pos = y % Y_BITS;

        return one << y_bit_pos;
    }

    constexpr static bit_type first_mask = 1;
    static bit_type move_next_mask(bit_type mask) {
        return mask << 1;
    }
};

template <typename bit_type>
struct BitTile {};

struct BitTileCommon {
    template <std::size_t x_size, std::size_t y_size, typename bit_type>
    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        x = x % x_size;
        y = y % y_size;

        auto shift = y * x_size + x;
        
        constexpr bit_type one = static_cast<bit_type>(1) << (sizeof(bit_type) * 8 - 1);
        return one >> shift;
    }

    template <typename bit_type>
    static bit_type move_next_mask(bit_type mask) {
        return mask >> 1;
    }

    template <typename bit_type>
    constexpr static bit_type first_mask = static_cast<bit_type>(1) << (sizeof(bit_type) * 8 - 1);
};

template <>
struct BitTile<std::uint16_t> {
    using bit_type = std::uint16_t;

    constexpr static std::size_t X_BITS = 4;
    constexpr static std::size_t Y_BITS = 4;
    

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }
};

template <>
struct BitTile<std::uint32_t> {
    using bit_type = std::uint32_t;

    constexpr static std::size_t X_BITS = 8;
    constexpr static std::size_t Y_BITS = 4;

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }
};

template <>
struct BitTile<std::uint64_t> {
    using bit_type = std::uint64_t;

    constexpr static std::size_t X_BITS = 8;
    constexpr static std::size_t Y_BITS = 8;

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }
};

template <typename bit_col_type, typename policy = BitColumns<bit_col_type>>
// template <typename bit_col_type, typename policy = BitTile<bit_col_type>>
class BitColsGrid {
  public:
    using size_type = std::size_t;

    template <typename grid_cell_t>
    using Grid = infrastructure::Grid<2, grid_cell_t>;

    using ONE_CELL_STATE = bool;
    constexpr static ONE_CELL_STATE DEAD = 0;
    constexpr static ONE_CELL_STATE ALIVE = 1;

    BitColsGrid(size_type original_x_size, size_t original_y_size)
        : _x_size(original_x_size), _y_size(original_y_size) {
        _x_size = original_x_size / policy::X_BITS;
        _y_size = original_y_size / policy::Y_BITS;

        bit_cols_grid.resize(x_size() * y_size(), 0);
    }

    template <typename grid_cell_t>
    BitColsGrid(const Grid<grid_cell_t>& grid) {
        assert_dim_has_correct_size(grid);

        _x_size = grid.template size_in<0>() / policy::X_BITS;
        _y_size = grid.template size_in<1>() / policy::Y_BITS;

        bit_cols_grid.resize(x_size() * y_size(), 0);
        fill_grid(grid);
    }

  public:
    ONE_CELL_STATE get_value_at(std::size_t x, std::size_t y) const {
        bit_col_type col = get_bit_col_from_original_coords(x, y);
        auto bit_mask = policy::get_bit_mask_for(x, y);

        return (col & bit_mask) ? ALIVE : DEAD;
    }

    void set_value_at(std::size_t x, std::size_t y, ONE_CELL_STATE state) {
        auto col = get_bit_col_from_original_coords(x, y);
        
        auto bit_mask = policy::get_bit_mask_for(x, y);

        if (state == ALIVE) {
            col |= bit_mask;
        }
        else {
            col &= ~bit_mask;
        }
        
        set_bit_col_from_original_coords(x, y, col);
    }

    bit_col_type get_bit_col_from_original_coords(std::size_t x, std::size_t y) const {
        return get_bit_col(x / policy::X_BITS, y / policy::Y_BITS);
    }

    bit_col_type get_bit_col(std::size_t x, std::size_t y) const {
        return bit_cols_grid[idx(x, y)];
    }

    void set_bit_col_from_original_coords(std::size_t x, std::size_t y, bit_col_type bit_col) {
        set_bit_col(x / policy::X_BITS, y / policy::Y_BITS, bit_col);
    }

    void set_bit_col(std::size_t x, std::size_t y, bit_col_type bit_col) {
        bit_cols_grid[idx(x, y)] = bit_col;
    }

    std::string debug_print_words() {
        std::ostringstream result;

        for (auto&& col : bit_cols_grid) {
            result << col << " ";
        }

        return result.str();
    }

    std::string debug_print(std::size_t line_limit = std::numeric_limits<std::size_t>::max()) {
        std::ostringstream result;

        for (std::size_t y = 0; y < original_y_size(); ++y) {
            for (std::size_t x = 0; x < original_x_size(); ++x) {
                auto val = get_value_at(x, y) ? '1' : '0';
                result << color_0_1(val) << " ";
            }
            result << "\n";

            if (y + 1 >= line_limit) {
                return result.str();
            }
        }

        return result.str();
    }

    

    size_type x_size() const {
        return _x_size;
    }

    size_type y_size() const {
        return _y_size;
    }

    size_type size() const {
        return x_size() * y_size();
    }

    size_type original_x_size() const {
        return _x_size * policy::X_BITS;
    }

    size_type original_y_size() const {
        return _y_size * policy::Y_BITS;
    }

    bit_col_type* data() {
        return bit_cols_grid.data();
    }

    std::vector<bit_col_type>* data_vector() {
        return &bit_cols_grid;
    }

    template <typename grid_cell_t>
    Grid<grid_cell_t> to_grid() const {
        auto _original_x_size = original_x_size();
        auto _original_y_size = original_y_size();

        Grid<grid_cell_t> grid(_original_x_size, _original_y_size);
        auto raw_data = grid.data();

        // for (std::size_t y = 0; y < _original_y_size; ++y) {
        //     for (std::size_t x = 0; x < _original_x_size; ++x) {
        //         auto val = get_value_at(x, y) ? 1 : 0;
        //         raw_data[in_grid_idx(x, y)] = static_cast<grid_cell_t>(val);
        //     }
        // }

        for (size_type y = 0; y < _original_y_size; y += policy::Y_BITS) {
            for (size_type x = 0; x < _original_x_size; x += policy::X_BITS) {

                auto col = get_bit_col_from_original_coords(x, y);
                auto mask = policy::first_mask;

                for (size_type y_bit = 0; y_bit < policy::Y_BITS; ++y_bit) {
                    for (size_type x_bit = 0; x_bit < policy::X_BITS; ++x_bit) {

                        auto value = (col & mask) ? 1 : 0;

                        raw_data[in_grid_idx(x + x_bit,y + y_bit)] = static_cast<grid_cell_t>(value);
                        
                        mask = policy::move_next_mask(mask);
                    }
                }
            }
        }


        return grid;
    }

  private:
    std::size_t in_grid_idx(std::size_t x, std::size_t y) const {
        return y * original_x_size() + x;
    }

    template <typename grid_cell_t>
    void assert_dim_has_correct_size(const Grid<grid_cell_t>& grid) {
        if (grid.template size_in<1>() % policy::Y_BITS != 0) {
            throw std::invalid_argument("Grid dimensions Y must be a multiple of " + std::to_string(policy::Y_BITS));
        }
        if (grid.template size_in<0>() % policy::X_BITS != 0) {
            throw std::invalid_argument("Grid dimensions X must be a multiple of " + std::to_string(policy::X_BITS));
        }
    }

    template <typename grid_cell_t>
    void fill_grid(const Grid<grid_cell_t>& grid) {
        auto _original_x_size = original_x_size();
        auto _original_y_size = original_y_size();

        auto raw_data = grid.data();

        // for (std::size_t y = 0; y < _original_y_size; ++y) {
        //     for (std::size_t x = 0; x < _original_x_size; ++x) {
        //         auto val = raw_data[in_grid_idx(x, y)];
        //         set_value_at(x, y, val);
        //     }
        // }

        for (size_type y = 0; y < _original_y_size; y += policy::Y_BITS) {
            for (size_type x = 0; x < _original_x_size; x += policy::X_BITS) {

                bit_col_type col = 0;
                auto bit_setter = policy::first_mask;
                
                for (size_type y_bit = 0; y_bit < policy::Y_BITS; ++y_bit) {
                    for (size_type x_bit = 0; x_bit < policy::X_BITS; ++x_bit) {

                        bit_col_type value = raw_data[in_grid_idx(x + x_bit,y + y_bit)] ? 1 : 0;

                        if (value) {
                            col |= bit_setter;
                        }
                        
                        bit_setter = policy::move_next_mask(bit_setter);
                    }
                }

                set_bit_col_from_original_coords(x, y, col);
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
        return y * x_size() + x;
    }

    std::vector<bit_col_type> bit_cols_grid;
    size_type _x_size;
    size_type _y_size;
};

} // namespace algorithms

#endif // GOL_BIT_COL_GRID_HPP