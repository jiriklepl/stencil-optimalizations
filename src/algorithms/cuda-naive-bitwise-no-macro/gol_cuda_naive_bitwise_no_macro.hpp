#ifndef GOL_CUDA_NAIVE_BITWISE_NO_MACRO_HPP
#define GOL_CUDA_NAIVE_BITWISE_NO_MACRO_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "models.hpp"
#include <cstddef>
#include <memory>

namespace algorithms {

template <std::size_t Bits>
class GoLCudaNaiveBitwiseNoMacro : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaiveBitwiseNoMacro() {};

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, char>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;
    
    using idx_t = std::int64_t;

    template <typename col_type>
    using BitGridCudaData = BitGridOnCudaWitOriginalSizes<col_type, idx_t>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        cuda_data.x_size_original = bit_grid->original_x_size();
        cuda_data.y_size_original = bit_grid->original_y_size();

        auto size = bit_grid->size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(col_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(col_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(col_type), cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        run_kernel(iterations);
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(col_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return bit_grid->to_grid();
    }

    
    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridCudaData<col_type> cuda_data;

    void run_kernel(size_type iterations);

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP