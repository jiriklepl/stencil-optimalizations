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

using StreamingDir = infrastructure::StreamingDirection;

template <typename grid_cell_t, std::size_t Bits>
class GoLCudaNaiveJustTiling : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
    GoLCudaNaiveJustTiling() {};

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        thread_block_size = this->params.thread_block_size;
        
        cuda_data.warp_dims = {
            .x = this->params.warp_dims_x,
            .y = this->params.warp_dims_y
        };
        
        cuda_data.warp_tile_dims = {
            .x = this->params.warp_tile_dims_x,
            .y = this->params.warp_tile_dims_y
        };

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
        switch (this->params.streaming_direction) {
            case StreamingDir::in_X:
                run_kernel<StreamingDir::in_X>(iterations);
                break;
            case StreamingDir::in_Y:
                run_kernel<StreamingDir::in_Y>(iterations);
                break;
            case StreamingDir::NAIVE:
                run_kernel<StreamingDir::NAIVE>(iterations);
                break;
            default:
                throw std::runtime_error("Invalid streaming direction");
        }
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(col_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridWithTiling<col_type> cuda_data;

    std::size_t thread_block_size;
    std::size_t _performed_iterations;


    template <StreamingDir Direction>
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