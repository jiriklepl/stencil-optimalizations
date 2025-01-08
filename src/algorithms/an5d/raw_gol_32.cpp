#include <stdlib.h>
#include <stdint.h>
#include "raw_gol.hpp"
#include "gol_ops_for_an5d.hpp"

#undef CELL_TYPE
#define CELL_TYPE ui32_t

void gol_32(CELL_TYPE* src, CELL_TYPE* dst, idx_t x_size, idx_t y_size, idx_t iters) {
    idx_t x_size_ext = x_size + 2;
    idx_t y_size_ext = y_size + 2;

    typedef CELL_TYPE (*c_grid_t)[y_size_ext][x_size_ext];

    idx_t grid_size_bytes = sizeof(CELL_TYPE) * x_size_ext * y_size_ext;
    void* working_memory = malloc(2 * grid_size_bytes);

    for (idx_t i = 0; i < 2 * grid_size_bytes; i++) {
        ((char*)working_memory)[i] = 0;
    }

    c_grid_t grid = (c_grid_t)working_memory;

    for (idx_t y = 0; y < y_size; y++) {
        for (idx_t x = 0; x < x_size; x++) {
            grid[0][y + 1][x + 1] = src[y * x_size + x];
        }
    }

    #pragma scop
    for (idx_t i = 0; i < iters; i++) {
        for (idx_t y = 1; y < y_size_ext - 1; y++) {
            for (idx_t x = 1; x < x_size_ext - 1; x++) {

                grid[(i + 1) % 2][y][x] = GOL_OP(
                    grid[i % 2][y - 1][x - 1],
                    grid[i % 2][y - 1][x],
                    grid[i % 2][y - 1][x + 1],
                    grid[i % 2][y][x - 1],
                    grid[i % 2][y][x],
                    grid[i % 2][y][x + 1],
                    grid[i % 2][y + 1][x - 1],
                    grid[i % 2][y + 1][x],
                    grid[i % 2][y + 1][x + 1]
                );

            }
        }
    }
    #pragma endscop

    for (idx_t y = 0; y < y_size; y++) {
        for (idx_t x = 0; x < x_size; x++) {
            dst[y * x_size + x] = grid[iters % 2][y + 1][x + 1];
        }
    }

    free(working_memory);
}
