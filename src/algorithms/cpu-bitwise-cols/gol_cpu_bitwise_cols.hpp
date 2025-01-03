#ifndef GOL_CPU_BITWISE_COLS_HPP
#define GOL_CPU_BITWISE_COLS_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"

namespace algorithms {

class GoLCpuBitwiseCols : public infrastructure::Algorithm<2, char> {
  public:
    using DataGrid = infrastructure::Grid<2, char>;
    using col_type = BitColsGrid::bit_col_type;
    using size_type = BitColsGrid::size_type;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = BitColsGrid(data);

        original_x_size = data.size_in<0>();
        original_y_size = data.size_in<1>();
    }

    void initialize_data_structures() override {
    }

    void run(size_type iterations) override {
        auto x_size = bit_grid.x_size();
        auto y_size = bit_grid.y_size();

        for (size_type i = 0; i < iterations; ++i) {
            for (size_type y = 0; y < y_size; ++y) {
                for (size_type x = 0; x < x_size; ++x) {
                    // clang-format off

                    col_type lt, ct, rt;
                    col_type lc, cc, rc; 
                    col_type lb, cb, rb;

                    load(bit_grid, x, y,
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb);

                    col_type new_center = BitwiseColsOps::compute_center_col(
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb
                    );

                    bit_grid.set_bit_col(x, y, new_center);

                    // clang-format on
                }
            }
        }
    }

    void finalize_data_structures() override {
        _result = DataGrid(original_x_size, original_y_size);

        for (size_type y = 0; y < original_y_size; ++y) {
            for (size_type x = 0; x < original_x_size; ++x) {
                auto col = bit_grid.get_bit_col(x, y / BitColsGrid::BITS_IN_COL);
                auto bit = y;

                _result[x][y] = ((col >> bit) & 1) ? 1 : 0;
            }
        }
    }

    DataGrid fetch_result() override {
        return _result;
    }

  private:
    // clang-format off
    void load(const BitColsGrid& bit_grid, size_type x, size_type y,
        col_type& lt, col_type& ct, col_type& rt,
        col_type& lc, col_type& cc, col_type& rc,
        col_type& lb, col_type& cb, col_type& rb) {

        load_one(bit_grid, lt, x - 1, y - 1);
        load_one(bit_grid, ct, x,     y - 1);
        load_one(bit_grid, rt, x + 1, y - 1);
        
        load_one(bit_grid, lc, x - 1, y    );
        load_one(bit_grid, cc, x,     y    );
        load_one(bit_grid, rc, x + 1, y    );
        
        load_one(bit_grid, lb, x - 1, y + 1);
        load_one(bit_grid, cb, x,     y + 1);
        load_one(bit_grid, rb, x + 1, y + 1);
    }
    // clang-format on

    void load_one(const BitColsGrid& bit_grid, col_type& col, size_type x, size_type y) {
        auto x_size = bit_grid.x_size();
        auto y_size = bit_grid.y_size();

        if (x < 0 || x >= x_size || y < 0 || y >= y_size) {
            col = 0;
        }
        else {
            col = bit_grid.get_bit_col(x, y);
        }
    }

    size_type original_x_size;
    size_type original_y_size;

    DataGrid _result;

    BitColsGrid bit_grid;
};

} // namespace algorithms

#endif // GOL_CPU_BITWISE_COLS_HPP