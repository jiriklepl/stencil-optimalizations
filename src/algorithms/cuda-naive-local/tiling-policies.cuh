#ifndef __TILING_POLICIES_CUH__
#define __TILING_POLICIES_CUH__

#include <stdexcept>
#include "../../infrastructure/experiment_params.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace algorithms {
using namespace algorithms::cuda_naive_local;

template <typename const_t, template <typename> class base_policy>
struct extended_policy {
    constexpr static const_t THREAD_BLOCK_SIZE = base_policy<const_t>::THREAD_BLOCK_SIZE;

    constexpr static const_t WARP_DIM_X = base_policy<const_t>::WARP_DIM_X;
    constexpr static const_t WARP_DIM_Y = base_policy<const_t>::WARP_DIM_Y;

    constexpr static const_t WARP_TILE_DIM_X = base_policy<const_t>::WARP_TILE_DIM_X;
    constexpr static const_t WARP_TILE_DIM_Y = base_policy<const_t>::WARP_TILE_DIM_Y;

    constexpr static const_t BLOCK_TILE_DIM_X = base_policy<const_t>::BLOCK_TILE_DIM_X;
    constexpr static const_t BLOCK_TILE_DIM_Y = base_policy<const_t>::BLOCK_TILE_DIM_Y;

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

    template <typename word_type, typename idx_t>
    static __device__ WarpInformation<word_type, idx_t> get_warp_info(
            idx_t x_size
        ) {
#ifdef __CUDA_ARCH__
        WarpInformation<word_type, idx_t> info;

        auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        info.warp_idx = idx / WARP_SIZE;
        info.lane_idx = idx % WARP_SIZE;
        
        auto x_block_count = x_size / ABSOLUTE_BLOCK_TILE_DIM_X;

        auto x_block = blockIdx.x % x_block_count;
        auto y_block = blockIdx.x / x_block_count;

        idx_t first_abs_x_in_block = x_block * ABSOLUTE_BLOCK_TILE_DIM_X;
        idx_t first_abs_y_in_block = y_block * ABSOLUTE_BLOCK_TILE_DIM_Y;

        idx_t warp_idx_within_block = threadIdx.x / WARP_SIZE;

        auto x_warp = warp_idx_within_block % BLOCK_TILE_DIM_X;
        auto y_warp = warp_idx_within_block / BLOCK_TILE_DIM_X;

        idx_t first_abs_x_in_warp = first_abs_x_in_block + x_warp * WARP_TILE_DIM_X;
        idx_t first_abs_y_in_warp = first_abs_y_in_block + y_warp * WARP_TILE_DIM_Y;

        idx_t x_within_warp = info.lane_idx % WARP_DIM_X;
        idx_t y_within_warp = info.lane_idx / WARP_DIM_X;

        info.x_abs_start = first_abs_x_in_warp + x_within_warp * X_COMPUTED_WORD_COUNT;
        info.y_abs_start = first_abs_y_in_warp + y_within_warp * Y_COMPUTED_WORD_COUNT;

        // if (idx == 0) {
        //     printf("x_size: %llu\n", x_size);
        //     printf("ABSOLUTE_BLOCK_TILE_DIM_X: %llu\n", ABSOLUTE_BLOCK_TILE_DIM_X);
        //     printf("ABSOLUTE_BLOCK_TILE_DIM_Y: %llu\n", ABSOLUTE_BLOCK_TILE_DIM_Y);
        //     printf("x_block_count: %llu\n", x_block_count);
        //     printf("x_block: %llu\n", x_block);
        //     printf("y_block: %llu\n", y_block);
        //     printf("first_abs_x_in_block: %llu\n", first_abs_x_in_block);
        //     printf("first_abs_y_in_block: %llu\n", first_abs_y_in_block);
        //     printf("warp_idx_within_block: %llu\n", warp_idx_within_block);
        //     printf("x_warp: %llu\n", x_warp);
        //     printf("y_warp: %llu\n", y_warp);
        //     printf("first_abs_x_in_warp: %llu\n", first_abs_x_in_warp);
        //     printf("first_abs_y_in_warp: %llu\n", first_abs_y_in_warp);
        //     printf("x_within_warp: %llu\n", x_within_warp);
        //     printf("y_within_warp: %llu\n", y_within_warp);
        //     printf("info.x_abs_start: %llu\n", info.x_abs_start);
        //     printf("info.y_abs_start: %llu\n", info.y_abs_start);
        // }

        return info;
#else
        WarpInformation<word_type, idx_t> info;
        
        printf("get_warp_info is not available on host\n");

        return info;
#endif
    }

};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x1 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 1;

    constexpr static const_t BLOCK_TILE_DIM_X = 2;
    constexpr static const_t BLOCK_TILE_DIM_Y = 4;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x4 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 4;

    constexpr static const_t BLOCK_TILE_DIM_X = 2;
    constexpr static const_t BLOCK_TILE_DIM_Y = 4;
};

} // namespace algorithms

#endif // __TILING_POLICIES_CUH__