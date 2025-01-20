#include "GOL.hpp"

#include <cuda_runtime.h>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

// texture<int,2> gridTex;
 
// __global__ void GOL(int dim, int *newGrid)
// {
//     int iy = blockDim.y * blockIdx.y + threadIdx.y;
//     int ix = blockDim.x * blockIdx.x + threadIdx.x;
//     int id = iy * dim + ix;
 
//     int numNeighbors;
 
//     float iyTex = (iy + 0.5f)/dim;
//     float ixTex = (ix + 0.5f)/dim;
//     float oneTex = 1.0f/dim;
 
//     if(iy < dim && ix < dim)
// {
//     //Get the number of neighbors for a given grid point
//     numNeighbors = tex2D(gridTex, iyTex+oneTex, ixTex) //upper/lower
//                  + tex2D(gridTex, iyTex-oneTex, ixTex)
//                  + tex2D(gridTex, iyTex, ixTex+oneTex) //right/left
//                  + tex2D(gridTex, iyTex, ixTex-oneTex)
//                  + tex2D(gridTex, iyTex-oneTex, ixTex-oneTex) //diagonals
//                  + tex2D(gridTex, iyTex-oneTex, ixTex+oneTex)
//                  + tex2D(gridTex, iyTex+oneTex, ixTex-oneTex) 
//                  + tex2D(gridTex, iyTex+oneTex, ixTex+oneTex);
 
//     int cell = tex2D(gridTex, iyTex, ixTex);
 
//     //Here we have explicitly all of the game rules
//     if (cell == 1 && numNeighbors < 2)
//         newGrid[id] = 0;
//     else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
//         newGrid[id] = 1;
//     else if (cell == 1 && numNeighbors > 3)
//         newGrid[id] = 0;
//     else if (cell == 0 && numNeighbors == 3)
//          newGrid[id] = 1;
//     else
//        newGrid[id] = cell;
 
// }
// }

template <typename grid_cell_t, BaselineVariant variant>
void GOL_Baseline<grid_cell_t, variant>::run_kernel_texture(size_type iterations) {
    (void) iterations;
    // dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    // int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
    // dim3 gridSize(linGrid,linGrid,1);

    // infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    // _performed_iterations = this->params.iterations;

    // for (size_type i = 0; i < iterations; ++i) {
    //     if (stop_watch.time_is_up()) {
    //         _performed_iterations = i;
    //         break;
    //     }

    //     if (i != 0) {
    //         std::swap(grid, new_grid);
    //     }

    //     GOL<<<gridSize, blockSize>>>(dim, grid, new_grid);
    //     CUCH(cudaPeekAtLastError());
    // }
}

}

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::Basic>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::SharedMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::TextureMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::Basic>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::SharedMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::TextureMemory>;

