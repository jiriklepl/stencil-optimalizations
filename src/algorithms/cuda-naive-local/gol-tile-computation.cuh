#ifndef GOL_TILE_COMPUTATION_CUH
#define GOL_TILE_COMPUTATION_CUH

#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "../../infrastructure/timer.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include "../_shared/common_grid_types.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include "x-generated_policies.hpp"
#include "./tiling-policies.cuh"
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace algorithms::cuda_naive_local {

using idx_t = std::int64_t;
// using idx_t = std::int32_t;
using shm_private_value_t = std::uint32_t;
using StreamingDir = infrastructure::StreamingDirection;

constexpr std::size_t WARP_SIZE = 32;
using WarpInfo = algorithms::cuda_naive_local::WarpInformation<idx_t>; 

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type, typename CudaData>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, CudaData&& data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename word_type, typename bit_grid_mode, typename tiling_policy, typename CudaData>
__device__ __forceinline__ bool compute_GOL_on_tile__naive_no_streaming(
    const WarpInfo& info, CudaData&& data) {
    
    bool tile_changed = false;

    auto x_fst = info.x_abs_start;
    auto x_lst = info.x_abs_start + tiling_policy::X_COMPUTED_WORD_COUNT;
    auto y_fst = info.y_abs_start;
    auto y_lst = info.y_abs_start + tiling_policy::Y_COMPUTED_WORD_COUNT;

    for (idx_t y = y_fst; y < y_lst; ++y) {
        for (idx_t x = x_fst; x < x_lst; ++x) {
    
            word_type lt = load<word_type>(x - 1, y - 1, data);
            word_type ct = load<word_type>(x + 0, y - 1, data);
            word_type rt = load<word_type>(x + 1, y - 1, data);
            word_type lc = load<word_type>(x - 1, y + 0, data);
            word_type cc = load<word_type>(x + 0, y + 0, data);
            word_type rc = load<word_type>(x + 1, y + 0, data);
            word_type lb = load<word_type>(x - 1, y + 1, data);
            word_type cb = load<word_type>(x + 0, y + 1, data);
            word_type rb = load<word_type>(x + 1, y + 1, data);
    
            auto new_cc = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb); 
            
            if (new_cc != cc) {
                tile_changed = true;
            }

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename word_type, typename bit_grid_mode, typename tiling_policy, typename CudaData>
__device__ __forceinline__ bool compute_GOL_on_tile__streaming_in_x(
    const WarpInfo& info, CudaData&& data) {
    
    bool tile_changed = false;

    auto x_fst = info.x_abs_start;
    auto x_lst = info.x_abs_start + tiling_policy::X_COMPUTED_WORD_COUNT;
    auto y_fst = info.y_abs_start;
    auto y_lst = info.y_abs_start + tiling_policy::Y_COMPUTED_WORD_COUNT;

    for (idx_t x = x_fst; x < x_lst; ++x) {
        idx_t y = y_fst;
        
        word_type lt, ct, rt;
        word_type lc, cc, rc;
        word_type lb, cb, rb;
        
        lc = load<word_type>(x - 1, y - 1, data);
        cc = load<word_type>(x + 0, y - 1, data);
        rc = load<word_type>(x + 1, y - 1, data);

        lb = load<word_type>(x - 1, y + 0, data);
        cb = load<word_type>(x + 0, y + 0, data);
        rb = load<word_type>(x + 1, y + 0, data);

        for (; y < y_lst; ++y) {

            lt = lc; 
            ct = cc; 
            rt = rc; 

            lc = lb; 
            cc = cb; 
            rc = rb; 
            
            lb = load<word_type>(x - 1, y + 1, data);
            cb = load<word_type>(x + 0, y + 1, data);
            rb = load<word_type>(x + 1, y + 1, data);
    
            auto new_cc = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb); 
            
            if (new_cc != cc) {
                tile_changed = true;
            }

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename word_type, typename bit_grid_mode, typename tiling_policy, typename CudaData>
__device__ __forceinline__ bool compute_GOL_on_tile__streaming_in_y(
    const WarpInfo& info, CudaData&& data) {
    
    bool tile_changed = false;

    auto x_fst = info.x_abs_start;
    auto x_lst = info.x_abs_start + tiling_policy::X_COMPUTED_WORD_COUNT;
    auto y_fst = info.y_abs_start;
    auto y_lst = info.y_abs_start + tiling_policy::Y_COMPUTED_WORD_COUNT;

    for (idx_t y = y_fst; y < y_lst; ++y) {
        idx_t x = x_fst;

        word_type lt, ct, rt;
        word_type lc, cc, rc;
        word_type lb, cb, rb;

        ct = load<word_type>(x - 1, y - 1, data);
        cc = load<word_type>(x - 1, y + 0, data);
        cb = load<word_type>(x - 1, y + 1, data);

        rt = load<word_type>(x + 0, y - 1, data);
        rc = load<word_type>(x + 0, y + 0, data);
        rb = load<word_type>(x + 0, y + 1, data);

        for (; x < x_lst; ++x) {

            lt = ct;
            lc = cc;
            lb = cb;

            ct = rt;
            cc = rc;
            cb = rb;

            rt = load<word_type>(x + 1, y - 1, data);
            rc = load<word_type>(x + 1, y + 0, data);
            rb = load<word_type>(x + 1, y + 1, data);
            
            auto new_cc = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb); 
            
            if (new_cc != cc) {
                tile_changed = true;
            }

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename word_type, typename bit_grid_mode, StreamingDir DIRECTION, typename tiling_policy, typename CudaData>
__device__ __forceinline__ bool compute_GOL_on_tile(
    const WarpInfo& info, CudaData&& data) {

    if constexpr (DIRECTION == StreamingDir::in_X) {
        return compute_GOL_on_tile__streaming_in_x<word_type, bit_grid_mode, tiling_policy>(info, data);
    }
    else if constexpr (DIRECTION == StreamingDir::in_Y) {
        return compute_GOL_on_tile__streaming_in_y<word_type, bit_grid_mode, tiling_policy>(info, data);
    }
    else if constexpr (DIRECTION == StreamingDir::NAIVE) {
        return compute_GOL_on_tile__naive_no_streaming<word_type, bit_grid_mode, tiling_policy>(info, data);
    }
    else {
        printf("Invalid streaming direction %d\n", DIRECTION);
    }
}

} // namespace algorithms::cuda_naive_local

#endif