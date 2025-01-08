#ifndef AN5D_GOL_32_64
#define AN5D_GOL_32_64

typedef unsigned int ui32_t;
typedef unsigned long long ui64_t;
typedef unsigned long long idx_t;

// dummy Game of Life operation -- redefine this
// #define GOL_OP(lt, tt, rt, lc, cc, rc, lb, cb, rb) \
//     lt + tt + rt + lc + rc + lb + cb + rb + cc

#define CELL_TYPE ui32_t

void gol_32(CELL_TYPE* src, CELL_TYPE* dst, idx_t x_size, idx_t y_size, idx_t iters);

#undef CELL_TYPE
#define CELL_TYPE ui64_t

void gol_64(CELL_TYPE* src, CELL_TYPE* dst, idx_t x_size, idx_t y_size, idx_t iters);

#endif