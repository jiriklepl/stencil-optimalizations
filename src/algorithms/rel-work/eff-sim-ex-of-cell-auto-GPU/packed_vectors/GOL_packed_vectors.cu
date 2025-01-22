#include "GOL.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

template <typename policy, typename CELL_TYPE>
__global__ void GOL_packed_vectors (int GRID_SIZE, CELL_TYPE *grid, CELL_TYPE *newGrid)
{
    constexpr int ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;
    constexpr unsigned int twos = 0x02020202U;
    constexpr unsigned int threes = 0x03030303U;
    constexpr unsigned int ones = 0x01010101U;

    const int ROW_SIZE = GRID_SIZE / ELEMENTS_PER_CELL;

    constexpr unsigned int shift_next = (ELEMENTS_PER_CELL-1)*8;

    // We want id ∈ [1,SIZE]
    const int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    const int id = iy * (ROW_SIZE+2) + ix;

    unsigned int numNeighbors = 0;
    if (iy>0 && iy <= GRID_SIZE && ix> 0 && ix <= ROW_SIZE) {
        const auto up_cell = grid[id-(ROW_SIZE+2)];
        numNeighbors = __vaddus4(numNeighbors, up_cell >> 8);
        numNeighbors = __vaddus4(numNeighbors, up_cell);
        numNeighbors = __vaddus4(numNeighbors, up_cell << 8);

        const auto down_cell = grid[id+(ROW_SIZE+2)];
        numNeighbors = __vaddus4(numNeighbors, down_cell >> 8);
        numNeighbors = __vaddus4(numNeighbors, down_cell);
        numNeighbors = __vaddus4(numNeighbors, down_cell << 8);

        const auto left_cell = grid[id+1] << shift_next;
        numNeighbors = __vaddus4(numNeighbors, left_cell);
        const auto upleft_cell = grid[id-(ROW_SIZE+1)] << shift_next;
        numNeighbors = __vaddus4(numNeighbors, upleft_cell);
        const auto downleft_cell = grid[id+(ROW_SIZE+3)] << shift_next;
        numNeighbors = __vaddus4(numNeighbors, downleft_cell);

        const auto right_cell = grid[id-1] >> shift_next;
        numNeighbors = __vaddus4(numNeighbors, right_cell);
        const auto upright_cell = grid[id-(ROW_SIZE+3)] >> shift_next;
        numNeighbors = __vaddus4(numNeighbors, upright_cell);
        const auto downright_cell = grid[id+(ROW_SIZE+1)] >> shift_next;
        numNeighbors = __vaddus4(numNeighbors, downright_cell);

        // fill a vectors filled with 2 and 3
        const auto cell = grid[id];
        numNeighbors = __vaddus4(numNeighbors, cell >> 8);
        numNeighbors = __vaddus4(numNeighbors, cell << 8);

        const auto alive_rule = __vcmpeq4(numNeighbors, twos) & cell;
        const auto general_rule = __vcmpeq4(numNeighbors, threes) & ones;

        // Copy new_cell to newGrid:
        newGrid[id] = alive_rule | general_rule;
    }
}

} // namespace

template <typename grid_cell_t, typename policy>
void GOL_Packed_vectors<grid_cell_t, policy>::run_kernel(size_type iterations) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGridx = (int)ceil(ROW_SIZE/(float)BLOCK_SIZE);
    int linGridy = (int)ceil(GRID_SIZE/(float)BLOCK_SIZE);
    dim3 gridSize(linGridx,linGridy,1);
 
    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }
        
        if (i != 0) {
            std::swap(grid, new_grid);
        }

        GOL_packed_vectors<policy><<<gridSize, blockSize>>>(GRID_SIZE, grid, new_grid);
        CUCH(cudaPeekAtLastError());
    }
}


} // namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy_vectors>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::INT, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy_vectors>;
