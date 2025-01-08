#ifndef GOL_AN5D_CPU_HPP
#define GOL_AN5D_CPU_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "raw_gol.hpp"

namespace algorithms {

template <std::size_t Bits>
class An5dCpu : public infrastructure::Algorithm<2, char> {
  public:
    An5dCpu() {};

    using col_type = typename BitsConst<Bits>::col_type;
    using BitGrid = algorithms::BitColsGrid<col_type>;
    using DataGrid = infrastructure::Grid<2, char>;
    using size_type = BitGrid::size_type;

    void set_and_format_input_data(const DataGrid& data) override {
        input_bit_grid = std::make_unique<BitGrid>(data);

        original_x_size = data.size_in<0>();
        original_y_size = data.size_in<1>();
    }

    void initialize_data_structures() override {
        result_bit_grid = std::make_unique<BitGrid>(original_x_size, original_y_size);
    }

    void run(size_type iterations) override;

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return result_bit_grid->to_grid();
    }

  private:

    size_type original_x_size;
    size_type original_y_size;

    DataGrid _result;

    std::unique_ptr<BitGrid> input_bit_grid;
    std::unique_ptr<BitGrid> result_bit_grid;
};


} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP