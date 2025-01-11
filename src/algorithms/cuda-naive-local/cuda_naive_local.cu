#ifndef CUDA_NAIVE_LOCAL_CU
#define CUDA_NAIVE_LOCAL_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_local.hpp"
#include <cuda_runtime.h>

namespace algorithms::cuda_naive_local {

using idx_t = std::int64_t;
// using idx_t = std::int32_t;

template <typename col_type, typename state_store_type>
using WarpInfo = algorithms::cuda_naive_local::WarpInformation<col_type, state_store_type, idx_t>; 

__device__ inline idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type, typename state_store_type>
__device__ inline idx_t x_tiles(BitGridWithChangeInfo<col_type, state_store_type> data) {
    return data.x_size / data.warp_tile_dims.x;
}

template <typename col_type, typename state_store_type>
__device__ inline idx_t y_tiles(const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    return data.y_size / data.warp_tile_dims.y;
}

template <typename col_type, typename state_store_type>
__device__ inline WarpInfo<col_type, state_store_type> get_warp_info(const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    WarpInfo<col_type, state_store_type> info;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    info.warp_idx = idx / 32;
    info.lane_idx = idx % 32;
    
    info.x_tile = info.warp_idx % x_tiles(data);
    info.y_tile = info.warp_idx / x_tiles(data);
    
    info.x_in_warp = info.lane_idx % data.warp_dims.x;
    info.y_in_warp = info.lane_idx / data.warp_dims.x;
    
    info.x_cols_in_warp = data.warp_tile_dims.x / data.warp_dims.x;
    info.y_rows_in_warp = data.warp_tile_dims.y / data.warp_dims.y;

    info.x_start = (info.x_tile * data.warp_tile_dims.x) + (info.x_in_warp * info.x_cols_in_warp);
    info.y_start = (info.y_tile * data.warp_tile_dims.y) + (info.y_in_warp * info.y_rows_in_warp);

    return info;
}

template <typename col_type, typename state_store_type>
__device__ inline col_type load(idx_t x, idx_t y, BitGridWithChangeInfo<col_type, state_store_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename col_type, typename state_store_type>
__device__ inline bool warp_tile_changed(
    idx_t x_tile, idx_t y_tile, 
    const WarpInfo<col_type, state_store_type>& info, const BitGridWithChangeInfo<col_type, state_store_type>& data) {

    return true;
    // return false;
}

template <typename col_type, typename state_store_type>
__device__ inline bool tile_or_neighbours_changed(
    idx_t x_tile, idx_t y_tile,
    const WarpInfo<col_type, state_store_type>& info, const BitGridWithChangeInfo<col_type, state_store_type>& data) {

    idx_t x_start = max(static_cast<idx_t>(0), x_tile - 1);
    idx_t y_start = max(static_cast<idx_t>(0), y_tile - 1);
    idx_t x_end = min(x_tiles(data), x_tile + 1);
    idx_t y_end = min(y_tiles(data), y_tile + 1);

    for(idx_t y = y_start; y < y_end; ++y) {
        for(idx_t x = x_start; x < x_end; ++x) {
            if (warp_tile_changed(x, y, info, data)) {
                return true;
            }
        }
    }

    return false;
}


template <typename col_type, typename state_store_type>
__device__ inline void cpy_to_output(
    const WarpInfo<col_type, state_store_type>& info, const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    
    for (idx_t y = info.y_start; y < info.y_start + info.y_rows_in_warp; ++y) {
        for (idx_t x = info.x_start; x < info.x_start + info.x_cols_in_warp; ++x) {
            data.output[get_idx(x, y, data.x_size)] = data.input[get_idx(x, y, data.x_size)];
        }
    }
}

template <typename col_type, typename state_store_type>
__device__ inline bool compute_GOL_on_tile(
    const WarpInfo<col_type, state_store_type>& info, const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    
    bool tile_changed = false;

    for (idx_t y = info.y_start; y < info.y_start + info.y_rows_in_warp; ++y) {
        idx_t x = info.x_start;

        col_type lt, ct, rt;
        col_type lc, cc, rc;
        col_type lb, cb, rb;

        ct = load(x - 1, y - 1, data);
        cc = load(x - 1, y + 0, data);
        cb = load(x - 1, y + 1, data);

        rt = load(x + 0, y - 1, data);
        rc = load(x + 0, y + 0, data);
        rb = load(x + 0, y + 1, data);

        for (; x < info.x_start + info.x_cols_in_warp; ++x) {

            lt = ct;
            lc = cc;
            lb = cb;

            ct = rt;
            cc = rc;
            cb = rb;

            rt = load(x + 1, y - 1, data);
            rc = load(x + 1, y + 0, data);
            rb = load(x + 1, y + 1, data);
            
            auto new_cc = CudaBitwiseOps<col_type>::compute_center_col(lt, ct, rt, lc, cc, rc, lb, cb, rb); 
            
            tile_changed |= new_cc != cc;

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename col_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<col_type, state_store_type> data) {
    auto info = get_warp_info(data);

    if (!tile_or_neighbours_changed(info.x_tile, info.y_tile, info, data)) {
        cpy_to_output(info, data);
        return;
    }

    bool tile_changed = compute_GOL_on_tile(info, data);
}

template <std::size_t Bits, typename state_store_type>
void GoLCudaNaiveLocal<Bits, state_store_type>::run_kernel(size_type iterations) {
    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

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