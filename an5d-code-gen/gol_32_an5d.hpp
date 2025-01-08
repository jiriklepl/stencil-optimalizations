#ifndef AN5D_GOL_32
#define AN5D_GOL_32

typedef unsigned int ui32_t;
typedef unsigned long long ui64_t;

#define CELL_TYPE ui32_t
void gol_32(CELL_TYPE* src, CELL_TYPE* dst, int x_size, int y_size, int iters);

#undef CELL_TYPE
#define CELL_TYPE ui64_t
void gol_64(CELL_TYPE* src, CELL_TYPE* dst, int x_size, int y_size, int iters);

#endif