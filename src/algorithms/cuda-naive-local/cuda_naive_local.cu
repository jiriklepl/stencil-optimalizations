// computing: 63, 3
// computing: 60, 3
// computing: 62, 3
// computing: 59, 3
// computing: 61, 3
// computing: 56, 3
// computing: 58, 3
// computing: 57, 3
// computing: 64, 4
// computing: 61, 4
// computing: 62, 4
// computing: 64, 3
// computing: 63, 4
// computing: 60, 4
// computing: 57, 4
// computing: 58, 4
// computing: 59, 4
// computing: 56, 4

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
using shm_private_value_t = std::uint32_t;

constexpr std::size_t WARP_SIZE = 32;

template <typename col_type, typename state_store_type>
using WarpInfo = algorithms::cuda_naive_local::WarpInformation<col_type, state_store_type, idx_t>; 

__device__ inline idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type, typename state_store_type>
__device__ inline WarpInfo<col_type, state_store_type> get_warp_info(const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    WarpInfo<col_type, state_store_type> info;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    info.warp_idx = idx / WARP_SIZE;
    info.lane_idx = idx % WARP_SIZE;
    
    info.x_tiles = data.x_size / data.warp_tile_dims.x;
    info.y_tiles = data.y_size / data.warp_tile_dims.y;

    info.x_tile = info.warp_idx % info.x_tiles;
    info.y_tile = info.warp_idx / info.x_tiles;
    
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
__device__ inline idx_t store_idx(idx_t x_tile, idx_t y_tile, const WarpInfo<col_type, state_store_type>& info) {
    auto tile_idx = y_tile * info.x_tiles + x_tile;
    auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

    auto store_idx = tile_idx / warp_tiles_in_block;
    return store_idx;
}

template <typename col_type, typename state_store_type>
__device__ inline int store_bit_idx(idx_t x_tile, idx_t y_tile, const WarpInfo<col_type, state_store_type>& info) {
    auto tile_idx = y_tile * info.x_tiles + x_tile;
    auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

    return tile_idx % warp_tiles_in_block;
}

template <typename col_type, typename state_store_type>
__device__ inline bool warp_tile_changed(
    idx_t x_tile, idx_t y_tile, 
    const WarpInfo<col_type, state_store_type>& info, state_store_type* store) {

    auto word = store[store_idx(x_tile, y_tile, info)];
    auto mask = 1 << store_bit_idx(x_tile, y_tile, info);

    return word & mask;
}

template <typename col_type, typename state_store_type>
__device__ inline bool tile_or_neighbours_changed(
    idx_t x_tile, idx_t y_tile,
    const WarpInfo<col_type, state_store_type>& info, state_store_type* store) {

    // idx_t x_start = max(static_cast<idx_t>(0), x_tile - 1);
    // idx_t y_start = max(static_cast<idx_t>(0), y_tile - 1);
    // idx_t x_end = min(info.x_tiles, x_tile + 1);
    // idx_t y_end = min(info.y_tiles, y_tile + 1);
    
    auto changed = false;

    for(idx_t y = y_tile - 1; y <= y_tile + 1; ++y) {
        for(idx_t x = x_tile - 1; x <= x_tile + 1; ++x) {

            if (x < 0 || y < 0 || x >= info.x_tiles || y >= info.y_tiles) {
                continue;
            }

            auto nei_changed = warp_tile_changed(x, y, info, store);

            if (info.x_tile == 61 && info.y_tile == 2 && info.lane_idx == 0 && info.iter == 1) {
                int val = nei_changed ? 1 : 0;
                auto str_idx = store_idx(x, y, info);
                auto bit = store_bit_idx(x, y, info);
                printf("checking: %llu, %llu ~ %d at %llu bit %d \n", x, y, val, str_idx, bit);

            }

            if (nei_changed) {
                changed = true;
            }
            // if (warp_tile_changed(x, y, info, store)) {
            //     return true;
            // }
        }
    }
    return changed;
    // return false;
}

template <typename state_store_type>
__device__ inline void set_changed_state_for_block(
    shm_private_value_t* block_store,
    state_store_type* global_store) {
        
    auto tiles = blockDim.x / WARP_SIZE;
    state_store_type result = 0;

    for (int i = 0; i < tiles; ++i) {
        state_store_type val = block_store[i] ? 1 : 0;
        result |= val << i;
    }

    if (result != 0) {
        printf("block changed: %d\n", blockIdx.x);
        for (int i = 0; i < tiles; ++i) {
            printf("block_store[%d] = %d\n", i, block_store[i] ? 1 : 0);
        }
    }

    global_store[blockIdx.x] = result;
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
            
            if (new_cc != cc) {
                tile_changed = true;
            }

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename col_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<col_type, state_store_type> data, int iter) {

    extern __shared__ shm_private_value_t block_store[];

    auto info = get_warp_info(data);
    info.iter = iter;

    bool warp_tile_changed;

    if (!tile_or_neighbours_changed(info.x_tile, info.y_tile, info, data.change_state_store.last)) {
    // if (false) {
        cpy_to_output(info, data);
        warp_tile_changed = false;
        // printf("No change\n");
    }
    else {
        if (iter == 1 && info.lane_idx == 0) {
            printf("computing: %llu, %llu\n", info.x_tile, info.y_tile);
        }

        auto local_tile_changed = compute_GOL_on_tile(info, data);

        if (local_tile_changed) {
            printf("tile changed: %llu, %llu\n", info.x_tile, info.y_tile);
            printf(" -> in warp little tile: %llu, %llu\n", info.x_in_warp, info.y_in_warp);
        }

        warp_tile_changed = __any_sync(0xFF'FF'FF'FF, local_tile_changed);
    }

    // warp_tile_changed = true;

    __syncthreads();

    if (info.lane_idx == 0) {
    // if (true) {
        auto tiles_per_block = blockDim.x / WARP_SIZE;
        block_store[info.warp_idx % tiles_per_block] = warp_tile_changed ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        set_changed_state_for_block(block_store, data.change_state_store.current);
    }

    __syncthreads();

    if (info.warp_idx == 0 && info.lane_idx == 0) {
        // auto tiles = blockDim.x / WARP_SIZE;
        // printf("tiles per block: %d\n", tiles);
        
        // for (int i = 0; i < tiles; ++i) {
        //     printf("block_store[%d] = %d\n", i, block_store[i]);
        // }
    }
}

template <typename elem_type>
__global__ void all_are(elem_type* data, std::size_t len, elem_type value) {


    for (int i = 0; i < len; ++i) {
        if (data[i] != value) {
            printf("Not all are, diff at [%d] = %llu\n", i, data[i]);

            return;
        }
    }
    
}

template <std::size_t Bits, typename state_store_type>
void GoLCudaNaiveLocal<Bits, state_store_type>::run_kernel(size_type iterations) {
    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

    auto warp_tile_per_block = block_size / WARP_SIZE;
    auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

    for (std::size_t i = 0; i < iterations; ++i) {
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
            std::swap(cuda_data.change_state_store.current, cuda_data.change_state_store.last);
        }

        game_of_live_kernel<<<blocks, block_size, shm_size>>>(cuda_data, i);
        check_state_stores();
    }

    // state_store_type expected = 0;

    // for (std::size_t i = 0; i < block_size / 32; ++i) {
    //     expected |= static_cast<state_store_type>(1) << i;
    // }

    // expected = ~expected;

    // std::cout << "expected: " << expected << std::endl;
    // // std::cout << "bits of expected: " << std::bit_set<64>(expected) << std::endl;
    // std::cout << "tiles per block: " << block_size / 32 << std::endl;

    // all_are<<<1, 1>>>(cuda_data.change_state_store.current, blocks, expected);
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


    // if (info.lane_idx == 0) {
    //     set_warp_tile_changed(false, info.x_tile, info.y_tile, info, data.change_state_store.current);
    // }

    // auto my_store_expected = blockIdx.x;
    // auto my_store_computed = store_idx(info.x_tile, info.y_tile, info);

    // auto my_store_bit_expected = threadIdx.x / WARP_SIZE;
    // auto my_store_bit_computed = store_bit_idx(info.x_tile, info.y_tile, info);

    // if (my_store_computed != my_store_expected) {
    //     printf("store - Expected: %d, Computed: %d\n", my_store_expected, my_store_computed);
    // }

    // if (my_store_bit_computed != my_store_bit_expected) {
    //     printf("bit - Expected: %d, Computed: %d\n", my_store_bit_expected, my_store_bit_computed);
    // }