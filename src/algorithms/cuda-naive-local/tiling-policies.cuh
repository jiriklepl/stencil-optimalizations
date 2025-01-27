#ifndef __TILING_POLICIES_CUH__
#define __TILING_POLICIES_CUH__

#include <stdexcept>
#include "../../infrastructure/experiment_params.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace algorithms {
using namespace algorithms::cuda_naive_local;

template <std::size_t thread_block_size, typename const_t> 
struct block_dims {};

template <typename const_t, template <typename> class base_policy>
struct extended_policy {
    constexpr static const_t THREAD_BLOCK_SIZE = base_policy<const_t>::THREAD_BLOCK_SIZE;

    constexpr static const_t WARP_DIM_X = base_policy<const_t>::WARP_DIM_X;
    constexpr static const_t WARP_DIM_Y = base_policy<const_t>::WARP_DIM_Y;

    constexpr static const_t WARP_TILE_DIM_X = base_policy<const_t>::WARP_TILE_DIM_X;
    constexpr static const_t WARP_TILE_DIM_Y = base_policy<const_t>::WARP_TILE_DIM_Y;

    constexpr static const_t BLOCK_TILE_DIM_X = block_dims<THREAD_BLOCK_SIZE, const_t>::x;
    constexpr static const_t BLOCK_TILE_DIM_Y = block_dims<THREAD_BLOCK_SIZE, const_t>::y;

    constexpr static const_t ABSOLUTE_BLOCK_TILE_DIM_X = WARP_TILE_DIM_X * BLOCK_TILE_DIM_X;
    constexpr static const_t ABSOLUTE_BLOCK_TILE_DIM_Y = WARP_TILE_DIM_Y * BLOCK_TILE_DIM_Y;

    constexpr static const_t X_COMPUTED_WORD_COUNT = WARP_TILE_DIM_X / WARP_DIM_X;
    constexpr static const_t Y_COMPUTED_WORD_COUNT = WARP_TILE_DIM_Y / WARP_DIM_Y;

    constexpr static const_t WARP_SIZE = 32;

    static bool is_for(infrastructure::ExperimentParams& params) {
        return params.thread_block_size == THREAD_BLOCK_SIZE && 
            params.warp_dims_x == WARP_DIM_X && 
            params.warp_dims_y == WARP_DIM_Y && 
            params.warp_tile_dims_x == WARP_TILE_DIM_X && 
            params.warp_tile_dims_y == WARP_TILE_DIM_Y;
    }

    template <typename idx_t>
    static __device__ WarpInformation<idx_t> get_warp_info(
            idx_t x_size, idx_t y_size
        ) {
#ifdef __CUDA_ARCH__
        WarpInformation<idx_t> info;

        auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        auto lane_idx = idx % WARP_SIZE;
        
        info.x_block_count = x_size / ABSOLUTE_BLOCK_TILE_DIM_X;
        info.y_block_count = y_size / ABSOLUTE_BLOCK_TILE_DIM_Y;

        info.x_block = blockIdx.x % info.x_block_count;
        info.y_block = blockIdx.x / info.x_block_count;

        idx_t first_abs_x_in_block = info.x_block * ABSOLUTE_BLOCK_TILE_DIM_X;
        idx_t first_abs_y_in_block = info.y_block * ABSOLUTE_BLOCK_TILE_DIM_Y;

        idx_t warp_idx_within_block = threadIdx.x / WARP_SIZE;

        info.x_warp = warp_idx_within_block % BLOCK_TILE_DIM_X;
        info.y_warp = warp_idx_within_block / BLOCK_TILE_DIM_X;

        idx_t first_abs_x_in_warp = first_abs_x_in_block + info.x_warp * WARP_TILE_DIM_X;
        idx_t first_abs_y_in_warp = first_abs_y_in_block + info.y_warp * WARP_TILE_DIM_Y;

        idx_t x_within_warp = lane_idx % WARP_DIM_X;
        idx_t y_within_warp = lane_idx / WARP_DIM_X;

        info.x_abs_start = first_abs_x_in_warp + x_within_warp * X_COMPUTED_WORD_COUNT;
        info.y_abs_start = first_abs_y_in_warp + y_within_warp * Y_COMPUTED_WORD_COUNT;


        return info;
#else
        WarpInformation<idx_t> info;
        
        printf("get_warp_info is not available on host\n");

        return info;
#endif
    }

};

template <typename const_t>
struct block_dims<64, const_t> {
    constexpr static const_t x = 1;
    constexpr static const_t y = 2;
};

template <typename const_t>
struct block_dims<128, const_t> {
    constexpr static const_t x = 2;
    constexpr static const_t y = 2;
};

template <typename const_t>
struct block_dims<256, const_t> {
    constexpr static const_t x = 2;
    constexpr static const_t y = 4;
};

template <typename const_t>
struct block_dims<512, const_t> {
    constexpr static const_t x = 4;
    constexpr static const_t y = 4;
};

template <typename const_t>
struct block_dims<1024, const_t> {
    constexpr static const_t x = 4;
    constexpr static const_t y = 8;
};

template <typename const_t>
struct b_dims {
    const_t x;
    const_t y;
};

template <typename const_t>
struct runtime_block_dims {
    static b_dims<const_t> get(infrastructure::ExperimentParams& params) {
        if (params.thread_block_size == 64) {
            return b_dims{
                .x= block_dims<64,const_t>::x,
                .y= block_dims<64,const_t>::y};
        } else if (params.thread_block_size == 128) {
            return b_dims{
                .x= block_dims<128,const_t>::x,
                .y= block_dims<128,const_t>::y};
        } else if (params.thread_block_size == 256) {
            return b_dims{
                .x= block_dims<256,const_t>::x,
                .y= block_dims<256,const_t>::y};
        } else if (params.thread_block_size == 512) {
            return b_dims{
                .x= block_dims<512,const_t>::x,
                .y= block_dims<512,const_t>::y};
        } else if (params.thread_block_size == 1024) {
            return b_dims{
                .x= block_dims<1024,const_t>::x,
                .y= block_dims<1024,const_t>::y};
        } else {
            throw std::runtime_error("Invalid thread block size");
        }
    }
};

} // namespace algorithms

#endif // __TILING_POLICIES_CUH__