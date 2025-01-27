#ifndef CUDA_NAIVE_LOCAL_CU
#define CUDA_NAIVE_LOCAL_CU

#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "../../infrastructure/timer.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_local.hpp"
#include "gol_cuda_naive_just_tiling.hpp"
#include <cuda_runtime.h>
#include "../_shared/common_grid_types.hpp"
#include "../_shared/common_grid_types.hpp"
#include "./tiling-policies.cuh"

namespace algorithms::cuda_naive_local {

using idx_t = std::int64_t;
// using idx_t = std::int32_t;
using shm_private_value_t = std::uint32_t;
using StreamingDir = infrastructure::StreamingDirection;


constexpr std::size_t WARP_SIZE = 32;

using WarpInfo = algorithms::cuda_naive_local::WarpInformation<idx_t>; 

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type, typename CudaData>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, CudaData&& data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

// template <typename word_type>
// __device__ __forceinline__ idx_t store_idx(idx_t x_tile, idx_t y_tile, const WarpInfo<word_type>& info) {
//     auto tile_idx = y_tile * info.x_tiles + x_tile;
//     auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

//     auto store_idx = tile_idx / warp_tiles_in_block;
//     return store_idx;
// }

// template <typename word_type>
// __device__ __forceinline__ int store_bit_idx(idx_t x_tile, idx_t y_tile, const WarpInfo<word_type>& info) {
//     auto tile_idx = y_tile * info.x_tiles + x_tile;
//     auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

//     return tile_idx % warp_tiles_in_block;
// }

// template <typename word_type, typename state_store_type>
// __device__ __forceinline__ bool warp_tile_changed(
//     idx_t x_tile, idx_t y_tile, 
//     const WarpInfo<word_type>& info, state_store_type* store) {

//     auto word = store[store_idx(x_tile, y_tile, info)];
//     auto mask = 1 << store_bit_idx(x_tile, y_tile, info);

//     return word & mask;
// }

// template <typename word_type, typename state_store_type>
// __device__ __forceinline__ bool tile_or_neighbours_changed(
//     idx_t x_tile, idx_t y_tile,
//     const WarpInfo<word_type>& info, state_store_type* store) {

//     idx_t x_start = max(static_cast<idx_t>(0), x_tile - 1);
//     idx_t y_start = max(static_cast<idx_t>(0), y_tile - 1);
//     idx_t x_end = min(info.x_tiles, x_tile + 1);
//     idx_t y_end = min(info.y_tiles, y_tile + 1);
    
//     for(idx_t y = y_start; y <= y_end + 1; ++y) {
//         for(idx_t x = x_start; x <= x_end; ++x) {
//             if (warp_tile_changed(x, y, info, store)) {
//                 return true;
//             }
//         }
//     }

//     return false;
// }

// template <typename state_store_type>
// __device__ __forceinline__ void set_changed_state_for_block(
//     shm_private_value_t* block_store,
//     state_store_type* global_store) {
        
//     auto tiles = blockDim.x / WARP_SIZE;
//     state_store_type result = 0;

//     for (int i = 0; i < tiles; ++i) {
//         state_store_type val = block_store[i] ? 1 : 0;
//         result |= val << i;
//     }

//     global_store[blockIdx.x] = result;
// }

template <typename tiling_policy, typename word_type, typename state_store_type>
__device__ __forceinline__ void cpy_to_output(
    const WarpInfo& info, const BitGridWithChangeInfo<word_type, state_store_type>& data) {

    auto x_fst = info.x_abs_start;
    auto x_lst = info.x_abs_start + tiling_policy::X_COMPUTED_WORD_COUNT;
    auto y_fst = info.y_abs_start;
    auto y_lst = info.y_abs_start + tiling_policy::Y_COMPUTED_WORD_COUNT;

    for (idx_t x = x_fst; x < x_lst; ++x) {
        for (idx_t y = y_fst; y < y_lst; ++y) {
            data.output[get_idx(x, y, data.x_size)] = data.input[get_idx(x, y, data.x_size)];
        }
    }
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


template <typename state_store_type>
__device__ __forceinline__ void set_changed_state_for_block(
    const WarpInfo& info, shm_private_value_t* block_store, state_store_type* global_store) {
        
    auto tiles = blockDim.x / WARP_SIZE;
    state_store_type result = 0;

    for (int i = 0; i < tiles; ++i) {
        state_store_type val = block_store[i] ? 1 : 0;
        result |= val << i;
    }

    global_store[blockIdx.x] = result;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ bool warp_tile_changed(
    idx_t x_tile, idx_t y_tile, 
    const WarpInfo& info, state_store_type* cached_store) {

    x_tile += tiling_policy::BLOCK_TILE_DIM_X;
    y_tile += tiling_policy::BLOCK_TILE_DIM_Y;

    auto x_word_idx = x_tile / tiling_policy::BLOCK_TILE_DIM_X;
    auto y_word_idx = y_tile / tiling_policy::BLOCK_TILE_DIM_Y;

    auto word = cached_store[y_word_idx * StateStoreInfo<state_store_type>::CACHE_SIZE_X + x_word_idx];

    auto x_bit_idx = x_tile % tiling_policy::BLOCK_TILE_DIM_X;
    auto y_bit_idx = y_tile % tiling_policy::BLOCK_TILE_DIM_Y;

    state_store_type one = 1;
    auto mask = one << (y_bit_idx * tiling_policy::BLOCK_TILE_DIM_X + x_bit_idx);

    return word & mask;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ bool tile_or_neighbours_changed(
    idx_t x_tile, idx_t y_tile,
    const WarpInfo& info, state_store_type* cached_store) {

    for(idx_t y = y_tile - 1; y <= y_tile + 1; ++y) {
        for(idx_t x = x_tile - 1; x <= x_tile + 1; ++x) {
            if (warp_tile_changed<tiling_policy>(x, y, info, cached_store)) {
                return true;
            }
        }
    }

    return false;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ void load_state_store(
    const WarpInfo& info,
    state_store_type* store, state_store_type* shared_store) {
    
    auto idx = threadIdx.x % StateStoreInfo<state_store_type>::CACHE_SIZE;
    
    auto x_block_to_load = idx % StateStoreInfo<state_store_type>::CACHE_SIZE_X + info.x_block - 1;
    auto y_block_to_load = idx / StateStoreInfo<state_store_type>::CACHE_SIZE_X + info.y_block - 1;

    // TODO this might fail
    if (x_block_to_load >= info.x_block_count || y_block_to_load >= info.y_block_count) {
        shared_store[idx] = 0;
        return;
    }

    auto store_idx = y_block_to_load * info.x_block_count + x_block_to_load;

    shared_store[idx] = store[store_idx];
}

// template <typename tiling_policy, typename state_store_type, typename CudaData>
// __device__ __forceinline__ void preload_caches(
//     const WarpInfo& info, const CudaData&& data,
//     state_store_type* cache_last, state_store_type* cache_before_last) {
    
//     if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE) {
//         load_state_store<tiling_policy>(info, data.change_state_store.before_last, cache_before_last);
//     }
//     else if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE * 2) {
//         load_state_store<tiling_policy>(info, data.change_state_store.last, cache_last);
//     }
// }

template <StreamingDir DIRECTION, typename bit_grid_mode, typename tiling_policy, typename word_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<word_type, state_store_type> data) {

    extern __shared__ shm_private_value_t block_store[];
    __shared__ state_store_type change_state_last[StateStoreInfo<state_store_type>::CACHE_SIZE];
    __shared__ state_store_type change_state_before_last[StateStoreInfo<state_store_type>::CACHE_SIZE];
    
    WarpInfo info = tiling_policy::template get_warp_info<idx_t>(data.x_size, data.y_size);
    // preload_caches<tiling_policy>(info, data, change_state_last, change_state_before_last);
    
    if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE) {
        load_state_store<tiling_policy>(info, data.change_state_store.before_last, change_state_before_last);
    }
    else if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE * 2) {
        load_state_store<tiling_policy>(info, data.change_state_store.last, change_state_last);
    }

    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("cache content: \n%lu %lu %lu \n%lu %lu %lu \n%lu %lu %lu \n", 
            change_state_last[0], change_state_last[1], change_state_last[2], 
            change_state_last[3], change_state_last[4], change_state_last[5],
            change_state_last[6], change_state_last[7], change_state_last[8]);
    }

    bool entire_tile_changed;

    if (!tile_or_neighbours_changed<tiling_policy>(info.x_warp, info.y_warp, info, change_state_last)) {
    // if (false) {
        
        if (warp_tile_changed<tiling_policy>(info.x_warp, info.y_warp, info, change_state_before_last)) {
            cpy_to_output<tiling_policy>(info, data);
        }
        
        entire_tile_changed = false;
    }
    else {
        auto local_tile_changed = compute_GOL_on_tile<word_type, bit_grid_mode, DIRECTION, tiling_policy>(info, data);
        entire_tile_changed = __any_sync(0xFF'FF'FF'FF, local_tile_changed);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        block_store[threadIdx.x / WARP_SIZE] = entire_tile_changed ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        set_changed_state_for_block(info, block_store, data.change_state_store.current);
    }
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename state_store_type, typename bit_grid_mode>
template <StreamingDir DIRECTION, typename tiling_policy>
void GoLCudaNaiveLocalWithState<grid_cell_t, Bits, state_store_type, bit_grid_mode>::run_kernel(size_type iterations) {

    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

    auto warp_tile_per_block = block_size / WARP_SIZE;
    auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        cudaEventRecord(events[i], stream);

        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }

        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
            rotate_state_stores();      
        }

        game_of_live_kernel<DIRECTION, bit_grid_mode, tiling_policy><<<blocks, block_size, shm_size, stream>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }

    cudaEventRecord(events[_performed_iterations], stream);
}

// -----------------------------------------------------------------------------------------------
// JUST TILING KERNEL
// -----------------------------------------------------------------------------------------------

template <StreamingDir DIRECTION, typename bit_grid_mode, typename tiling_policy, typename word_type>
__global__ void game_of_live_kernel_just_tiling(BitGridWithTiling<word_type> data) {
    WarpInfo info = tiling_policy::template get_warp_info<idx_t>(data.x_size, data.y_size);
    compute_GOL_on_tile<word_type, bit_grid_mode, DIRECTION, tiling_policy>(info, data);
}


template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
template <StreamingDir DIRECTION, typename tiling_policy>
void GoLCudaNaiveJustTiling<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) {
    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }

        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel_just_tiling<DIRECTION, bit_grid_mode, tiling_policy><<<blocks, block_size>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint64_t, algorithms::BitTileMode>;

template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitWastefulRowsMode>;

#endif // CUDA_NAIVE_LOCAL_CU
