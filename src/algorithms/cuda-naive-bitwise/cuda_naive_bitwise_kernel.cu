#ifndef CUDA_NAIVE_KERNEL_BITWISE_CU
#define CUDA_NAIVE_KERNEL_BITWISE_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_bitwise.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"

namespace algorithms {

using idx_t = std::int64_t;

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type>
__device__ __forceinline__ col_type load(idx_t x, idx_t y, BitGridOnCuda<col_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename col_type>
__global__ void game_of_live_kernel(BitGridOnCuda<col_type> data) {
    idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= data.x_size || y >= data.y_size)
        return;

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

} // namespace

template <typename grid_cell_t, std::size_t Bits>
void GoLCudaNaiveBitwise<grid_cell_t, Bits>::run_kernel(size_type iterations) { // Added template parameter
    dim3 block = get_2d_block(this->params.thread_block_size);
    dim3 grid((cuda_data.x_size + block.x - 1) / block.x, (cuda_data.y_size + block.y - 1) / block.y);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            return;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<<<grid, block>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 16>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 32>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 64>;

template class algorithms::GoLCudaNaiveBitwise<common::INT, 16>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 32>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 64>;

#endif // CUDA_NAIVE_KERNEL_CU