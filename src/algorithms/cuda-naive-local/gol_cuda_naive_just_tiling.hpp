#ifndef GOL_CUDA_NAIVE_JUST_TILING_HPP
#define GOL_CUDA_NAIVE_JUST_TILING_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

namespace algorithms::cuda_naive_local {

template <std::size_t Bits>
class GoLCudaNaiveJustTiling : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaiveJustTiling() {};

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, char>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        cuda_data.warp_dims = { .x = 8, .y = 4};
        cuda_data.warp_tile_dims = { .x = 16, .y = 8};
        thread_block_size = 32 * 8;

        assert(warp_size() == 32);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

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

  private:
    BitGrid_ptr bit_grid;
    BitGridWithTiling<col_type> cuda_data;

    std::size_t thread_block_size;

    void run_kernel(size_type iterations);

    std::size_t warp_tile_size() const {
        return cuda_data.warp_tile_dims.x * cuda_data.warp_tile_dims.y;
    }

    std::size_t warp_size() const {
        return cuda_data.warp_dims.x * cuda_data.warp_dims.y;
    }

    std::size_t get_warp_tiles_count() {
        return bit_grid->size() / warp_tile_size();
    }

    std::size_t get_thread_block_count() {
        auto warps_per_block = thread_block_size / warp_size();
        auto computed_elems_in_block = warps_per_block * warp_tile_size(); 

        return bit_grid->size() / computed_elems_in_block;
    }

    std::size_t state_store_word_count() {
        return get_thread_block_count();
    }

    std::size_t tiles_per_block() {
        return thread_block_size / warp_size();
    }

};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP