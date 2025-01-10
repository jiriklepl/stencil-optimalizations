#ifndef CUDA_NAIVE_LOCAL_CU
#define CUDA_NAIVE_LOCAL_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_local.hpp"
#include <cuda_runtime.h>

namespace algorithms::cuda_naive_local {

using idx_t = std::int64_t;

__device__ inline idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type, typename state_store_type>
__device__ inline idx_t x_tiles(BitGridWithChangeInfo<col_type, state_store_type> data) {
    return data.x_size / data.warp_tile_dims.x;
}

template <typename col_type, typename state_store_type>
__device__ inline idx_t y_tiles(BitGridWithChangeInfo<col_type, state_store_type> data) {
    return data.y_size / data.warp_tile_dims.y;
}

template <typename col_type, typename state_store_type>
__device__ inline col_type load(idx_t x, idx_t y, BitGridWithChangeInfo<col_type, state_store_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename col_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<col_type, state_store_type> data) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    idx_t warp_idx = idx / 32;
    idx_t lane_idx = idx % 32;

    idx_t x_tile = warp_idx % x_tiles(data);
    idx_t y_tile = warp_idx / x_tiles(data);

    idx_t x_in_warp = lane_idx % data.warp_dims.x;
    idx_t y_in_warp = lane_idx / data.warp_dims.x;

    idx_t x_cols_in_warp = data.warp_tile_dims.x / data.warp_dims.x;
    idx_t y_rows_in_warp = data.warp_tile_dims.y / data.warp_dims.y;

    idx_t x_start = (x_tile * data.warp_tile_dims.x) + (x_in_warp * x_cols_in_warp);
    idx_t y_start = (y_tile * data.warp_tile_dims.y) + (y_in_warp * y_rows_in_warp);

    for (idx_t y = y_start; y < y_start + y_rows_in_warp; ++y) {
        for (idx_t x = x_start; x < x_start + x_cols_in_warp; ++x) {

            col_type lt = load(x - 1, y - 1, data);
            col_type ct = load(x, y - 1, data);
            col_type rt = load(x + 1, y - 1, data);

            col_type lc = load(x - 1, y, data);
            col_type cc = load(x, y, data);
            col_type rc = load(x + 1, y, data);

            col_type lb = load(x - 1, y + 1, data);
            col_type cb = load(x, y + 1, data);
            col_type rb = load(x + 1, y + 1, data);

            data.output[get_idx(x, y, data.x_size)] =
                CudaBitwiseOps<col_type>::compute_center_col(lt, ct, rt, lc, cc, rc, lb, cb, rb);
        }
    }

}

template <std::size_t Bits, typename state_store_type>
void GoLCudaNaiveLocal<Bits, state_store_type>::run_kernel(size_type iterations) {
    auto block_size = 32 * 4;
    auto blocks = get_thread_block_count(block_size);

    for (std::size_t i = 0; i < iterations; ++i) {
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<<<blocks, block_size>>>(cuda_data);
    }
}

} // namespace algorithms

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint64_t>;

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint64_t>;

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint64_t>;

#endif // CUDA_NAIVE_LOCAL_CU