#ifndef GOL_CUDA_NAIVE_LOCAL_HPP
#define GOL_CUDA_NAIVE_LOCAL_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace algorithms::cuda_naive_local {

template <std::size_t Bits, typename state_store_type = std::uint64_t>
class GoLCudaNaiveLocal : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaiveLocal() {};

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, char>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        cuda_data.warp_dims = { .x = 4, .y = 8};
        cuda_data.warp_tile_dims = { .x = 64, .y = 8};

        assert(warp_size() == 32);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();   

        // auto size_of_warp_tile = cuda_data.warp_dims.x * cuda_data.warp_dims.y;
        // auto change_state_store_size = size / ;

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

        cudaFree(cuda_data.input);
        cudaFree(cuda_data.output);
    }

    DataGrid fetch_result() override {
        return bit_grid->to_grid();
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridWithChangeInfo<col_type, state_store_type> cuda_data;

    void run_kernel(size_type iterations);

    std::size_t warp_tile_size() const {
        return cuda_data.warp_tile_dims.x * cuda_data.warp_tile_dims.y;
    }

    std::size_t warp_size() const {
        return cuda_data.warp_dims.x * cuda_data.warp_dims.y;
    }

    std::size_t get_thread_block_count(std::size_t thread_block_size) {
        auto warps_per_block = thread_block_size / warp_size();
        auto computed_elems_in_block = warps_per_block * warp_tile_size(); 

        std::cout << "computed_elems_in_block: " << computed_elems_in_block << std::endl;

        return bit_grid->size() / computed_elems_in_block;
    }
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP