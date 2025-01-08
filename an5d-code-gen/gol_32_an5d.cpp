#include <stdlib.h>
#include <stdint.h>
#include "cuda_gol_32_64_an5d.hpp"

#undef CELL_TYPE
#define CELL_TYPE ui32_t

CELL_TYPE GOL_OP(
    CELL_TYPE lt, CELL_TYPE tt, CELL_TYPE rt,
    CELL_TYPE lc, CELL_TYPE cc, CELL_TYPE rc,
    CELL_TYPE lb, CELL_TYPE cb, CELL_TYPE rb);

void cuda_gol_32(CELL_TYPE* src, CELL_TYPE* dst, int x_size, int y_size, int iters) {
    int x_size_ext = x_size + 2;
    int y_size_ext = y_size + 2;

    typedef CELL_TYPE (*c_grid_t)[y_size_ext][x_size_ext];

    int grid_size_bytes = sizeof(CELL_TYPE) * x_size_ext * y_size_ext;
    void* working_memory = malloc(2 * grid_size_bytes);

    for (int i = 0; i < 2 * grid_size_bytes; i++) {
        ((char*)working_memory)[i] = 0;
    }

    c_grid_t grid = (c_grid_t)working_memory;

    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            grid[0][y + 1][x + 1] = src[y * x_size + x];
        }
    }

    #pragma scop
    for (int i = 0; i < iters; i++) {
        for (int y = 1; y < y_size_ext - 1; y++) {
            for (int x = 1; x < x_size_ext - 1; x++) {

                grid[(i + 1) % 2][y][x] = GOL_OP(
                    grid[i % 2][y - 1][x - 1], grid[i % 2][y - 1][x    ], grid[i % 2][y - 1][x + 1],
                    grid[i % 2][y    ][x - 1], grid[i % 2][y    ][x    ], grid[i % 2][y    ][x + 1],
                    grid[i % 2][y + 1][x - 1], grid[i % 2][y + 1][x    ], grid[i % 2][y + 1][x + 1]
                );

            }
        }
    }
    #pragma endscop

    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            dst[y * x_size + x] = grid[iters % 2][y + 1][x + 1];
        }
    }

    free(working_memory);
}
