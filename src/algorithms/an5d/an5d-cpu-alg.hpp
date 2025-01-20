#ifndef GOL_AN5D_CPU_HPP
#define GOL_AN5D_CPU_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/bitwise/bitwise-ops/templated-cols.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace algorithms {

enum class ExecModel {
    CPU = 0,
    CUDA = 1,
};

template <typename grid_cell_t, std::size_t Bits, ExecModel Model>
class An5dAlg : public infrastructure::Algorithm<2, grid_cell_t> {
  public:
    An5dAlg() {};

    using word_type = typename BitsConst<Bits>::word_type;
    using BitGrid = algorithms::GeneralBitGrid<word_type>;
    
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using size_type = BitGrid::size_type;

    bool is_an5d_cuda_alg() const override {
        return Model == ExecModel::CUDA;
    }

    void set_and_format_input_data(const DataGrid& data) override {
        input_bit_grid = std::make_unique<BitGrid>(data);

        original_x_size = data.template size_in<0>();
        original_y_size = data.template size_in<1>();
    }

    void initialize_data_structures() override {
        result_bit_grid = std::make_unique<BitGrid>(original_x_size, original_y_size);
    }

    void run(size_type iterations) override;

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return result_bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return this->params.iterations;
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