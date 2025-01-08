#include "gol_64_an5d_kernel.hu"
typedef ui64_t (*c_grid_t)[y_size_ext][x_size_ext];
__device__ unsigned long long __sbref_wrap(unsigned long long *sb, size_t index) { return sb[index]; }

__global__ void kernel0_4(unsigned long long *grid, int x_size_ext, int y_size_ext, int iters, int c0)
{
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
    const AN5D_TYPE __c0Len = (iters - 0);
    const AN5D_TYPE __c0Pad = (0);
    #define __c0 c0
    const AN5D_TYPE __c1Len = (y_size_ext - 1 - 1);
    const AN5D_TYPE __c1Pad = (1);
    #define __c1 c1
    const AN5D_TYPE __c2Len = (x_size_ext - 1 - 1);
    const AN5D_TYPE __c2Pad = (1);
    #define __c2 c2
    const AN5D_TYPE __halo1 = 1;
    const AN5D_TYPE __halo2 = 1;
    const AN5D_TYPE __side0Len = 4;
    const AN5D_TYPE __side1Len = 128;
    const AN5D_TYPE __side2Len = 24;
    const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
    const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
    const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
    const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
    const AN5D_TYPE __blockSize = 1 * __side2LenOl;
    const AN5D_TYPE __side1Num = (__c1Len + __side1Len - 1) / __side1Len;
    const AN5D_TYPE __side2Num = (__c2Len + __side2Len - 1) / __side2Len;
    const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;
    const AN5D_TYPE __local_c2 = __tid;
    const AN5D_TYPE __c1Id = blockIdx.x / __side2Num;
    const AN5D_TYPE __c2 = (blockIdx.x % __side2Num) * __side2Len + __local_c2 + __c2Pad - __OlLen2;
    unsigned long long __reg_0;
    unsigned long long __reg_1_0;
    unsigned long long __reg_1_1;
    unsigned long long __reg_1_2;
    unsigned long long __reg_2_0;
    unsigned long long __reg_2_1;
    unsigned long long __reg_2_2;
    unsigned long long __reg_3_0;
    unsigned long long __reg_3_1;
    unsigned long long __reg_3_2;
    unsigned long long __reg_4_0;
    unsigned long long __reg_4_1;
    unsigned long long __reg_4_2;
    __shared__ unsigned long long __a_sb_double[__blockSize * 2];
    unsigned long long *__a_sb = __a_sb_double;
    const AN5D_TYPE __loadValid = 1 && __c2 >= __c2Pad - __halo2 && __c2 < __c2Pad + __c2Len + __halo2;
    const AN5D_TYPE __updateValid = 1 && __c2 >= __c2Pad && __c2 < __c2Pad + __c2Len;
    const AN5D_TYPE __writeValid1 = __updateValid && __local_c2 >= (__halo2 * 1) && __local_c2 < __side2LenOl - (__halo2 * 1);
    const AN5D_TYPE __writeValid2 = __updateValid && __local_c2 >= (__halo2 * 2) && __local_c2 < __side2LenOl - (__halo2 * 2);
    const AN5D_TYPE __writeValid3 = __updateValid && __local_c2 >= (__halo2 * 3) && __local_c2 < __side2LenOl - (__halo2 * 3);
    const AN5D_TYPE __writeValid4 = __updateValid && __local_c2 >= (__halo2 * 4) && __local_c2 < __side2LenOl - (__halo2 * 4);
    const AN5D_TYPE __storeValid = __writeValid4;
    AN5D_TYPE __c1;
    AN5D_TYPE __h;
    const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;
    #define __LOAD(reg, h) do { if (__loadValid) { __c1 = __c1Pad2 - __halo1 + h; reg = grid[((__c0 % 2) * y_size_ext + __c1) * x_size_ext + __c2]; }} while (0)
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_0_wrap(__rn0, __a) do { __rn0 = GOL_OP((__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_0(out, a) do { __CALCEXPR_0_wrap(out, a);  } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_1_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_1(out, a) do { unsigned long long etmp; __CALCEXPR_1_wrap(etmp, a); out += etmp; } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_2_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1))); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_2(out, a) do { unsigned long long etmp; __CALCEXPR_2_wrap(etmp, a); out += etmp; } while (0);
    #define __CALCEXPR(out0, out1, out2, reg) do { __CALCEXPR_0(out0, reg); __CALCEXPR_1(out1, reg); __CALCEXPR_2(out2, reg); } while (0);
    #define __CALC1(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid1) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC2(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid2) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC3(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid3) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC4(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid4) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __STORE(h, out) do { if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; __DEST = out; }} while (0)
    if (__c1Id == 0)
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_0);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_0);
      __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
      __LOAD(__reg_0, 5);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
      __STORE(1, __reg_4_1);
      __LOAD(__reg_0, 6);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
      __STORE(2, __reg_4_2);
      __LOAD(__reg_0, 7);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
      __STORE(3, __reg_4_0);
      __LOAD(__reg_0, 8);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
      __STORE(4, __reg_4_1);
    }
    else
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __LOAD(__reg_0, 5);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __LOAD(__reg_0, 6);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
      __LOAD(__reg_0, 7);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
      __LOAD(__reg_0, 8);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
      __STORE(4, __reg_4_1);
    }
    __a_sb = __a_sb_double + __blockSize * 0;
    if (__c1Id == __side1Num - 1)
    {
      for (__h = 9; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - 4;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 4, __reg_4_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h - 4, __reg_4_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
        __STORE(__h - 4, __reg_4_1);
        __h++;
      }
      if (0) {}
      else if (__h + 1 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_1, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 4, __reg_4_2);
        __reg_1_0 = __reg_0;
        __CALC2(__reg_2_1, __reg_2_1, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h - 3, __reg_4_0);
        __reg_2_0 = __reg_1_0;
        __CALC3(__reg_3_1, __reg_3_1, __reg_3_2, __reg_2_0);
        __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
        __STORE(__h - 2, __reg_4_1);
        __reg_3_0 = __reg_2_0;
        __CALC4(__reg_4_1, __reg_4_1, __reg_4_2, __reg_3_0);
        __STORE(__h - 1, __reg_4_2);
      }
      else if (__h + 2 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 4, __reg_4_2);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_2, __reg_1_2, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h - 3, __reg_4_0);
        __reg_1_1 = __reg_0;
        __CALC2(__reg_2_2, __reg_2_2, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
        __STORE(__h - 2, __reg_4_1);
        __reg_2_1 = __reg_1_1;
        __CALC3(__reg_3_2, __reg_3_2, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 1, __reg_4_2);
        __reg_3_1 = __reg_2_1;
        __CALC4(__reg_4_2, __reg_4_2, __reg_4_0, __reg_3_1);
        __STORE(__h + 0, __reg_4_0);
      }
      else if (__h + 3 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 4, __reg_4_2);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h - 3, __reg_4_0);
        __LOAD(__reg_0, __h + 2);
        __CALC1(__reg_1_0, __reg_1_0, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
        __STORE(__h - 2, __reg_4_1);
        __reg_1_2 = __reg_0;
        __CALC2(__reg_2_0, __reg_2_0, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 1, __reg_4_2);
        __reg_2_2 = __reg_1_2;
        __CALC3(__reg_3_0, __reg_3_0, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h + 0, __reg_4_0);
        __reg_3_2 = __reg_2_2;
        __CALC4(__reg_4_0, __reg_4_0, __reg_4_1, __reg_3_2);
        __STORE(__h + 1, __reg_4_1);
      }
    }
    else
    {
      for (__h = 9; __h <= __side1LenOl - 3;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
        __STORE(__h - 4, __reg_4_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
        __STORE(__h - 4, __reg_4_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
        __STORE(__h - 4, __reg_4_1);
        __h++;
      }
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __CALC4(__reg_4_1, __reg_4_0, __reg_4_2, __reg_3_0);
      __STORE(__h - 4, __reg_4_2);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __CALC4(__reg_4_2, __reg_4_1, __reg_4_0, __reg_3_1);
      __STORE(__h - 4, __reg_4_0);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __CALC4(__reg_4_0, __reg_4_2, __reg_4_1, __reg_3_2);
      __STORE(__h - 4, __reg_4_1);
      __h++;
    }
}
__global__ void kernel0_3(unsigned long long *grid, int x_size_ext, int y_size_ext, int iters, int c0)
{
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
    const AN5D_TYPE __c0Len = (iters - 0);
    const AN5D_TYPE __c0Pad = (0);
    #define __c0 c0
    const AN5D_TYPE __c1Len = (y_size_ext - 1 - 1);
    const AN5D_TYPE __c1Pad = (1);
    #define __c1 c1
    const AN5D_TYPE __c2Len = (x_size_ext - 1 - 1);
    const AN5D_TYPE __c2Pad = (1);
    #define __c2 c2
    const AN5D_TYPE __halo1 = 1;
    const AN5D_TYPE __halo2 = 1;
    const AN5D_TYPE __side0Len = 3;
    const AN5D_TYPE __side1Len = 128;
    const AN5D_TYPE __side2Len = 26;
    const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
    const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
    const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
    const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
    const AN5D_TYPE __blockSize = 1 * __side2LenOl;
    const AN5D_TYPE __side1Num = (__c1Len + __side1Len - 1) / __side1Len;
    const AN5D_TYPE __side2Num = (__c2Len + __side2Len - 1) / __side2Len;
    const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;
    const AN5D_TYPE __local_c2 = __tid;
    const AN5D_TYPE __c1Id = blockIdx.x / __side2Num;
    const AN5D_TYPE __c2 = (blockIdx.x % __side2Num) * __side2Len + __local_c2 + __c2Pad - __OlLen2;
    unsigned long long __reg_0;
    unsigned long long __reg_1_0;
    unsigned long long __reg_1_1;
    unsigned long long __reg_1_2;
    unsigned long long __reg_2_0;
    unsigned long long __reg_2_1;
    unsigned long long __reg_2_2;
    unsigned long long __reg_3_0;
    unsigned long long __reg_3_1;
    unsigned long long __reg_3_2;
    __shared__ unsigned long long __a_sb_double[__blockSize * 2];
    unsigned long long *__a_sb = __a_sb_double;
    const AN5D_TYPE __loadValid = 1 && __c2 >= __c2Pad - __halo2 && __c2 < __c2Pad + __c2Len + __halo2;
    const AN5D_TYPE __updateValid = 1 && __c2 >= __c2Pad && __c2 < __c2Pad + __c2Len;
    const AN5D_TYPE __writeValid1 = __updateValid && __local_c2 >= (__halo2 * 1) && __local_c2 < __side2LenOl - (__halo2 * 1);
    const AN5D_TYPE __writeValid2 = __updateValid && __local_c2 >= (__halo2 * 2) && __local_c2 < __side2LenOl - (__halo2 * 2);
    const AN5D_TYPE __writeValid3 = __updateValid && __local_c2 >= (__halo2 * 3) && __local_c2 < __side2LenOl - (__halo2 * 3);
    const AN5D_TYPE __storeValid = __writeValid3;
    AN5D_TYPE __c1;
    AN5D_TYPE __h;
    const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;
    #define __LOAD(reg, h) do { if (__loadValid) { __c1 = __c1Pad2 - __halo1 + h; reg = grid[((__c0 % 2) * y_size_ext + __c1) * x_size_ext + __c2]; }} while (0)
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_0_wrap(__rn0, __a) do { __rn0 = GOL_OP((__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_0(out, a) do { __CALCEXPR_0_wrap(out, a);  } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_1_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_1(out, a) do { unsigned long long etmp; __CALCEXPR_1_wrap(etmp, a); out += etmp; } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_2_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1))); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_2(out, a) do { unsigned long long etmp; __CALCEXPR_2_wrap(etmp, a); out += etmp; } while (0);
    #define __CALCEXPR(out0, out1, out2, reg) do { __CALCEXPR_0(out0, reg); __CALCEXPR_1(out1, reg); __CALCEXPR_2(out2, reg); } while (0);
    #define __CALC1(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid1) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC2(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid2) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC3(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid3) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __STORE(h, out) do { if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; __DEST = out; }} while (0)
    if (__c1Id == 0)
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_0);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __STORE(1, __reg_3_1);
      __LOAD(__reg_0, 5);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __STORE(2, __reg_3_2);
      __LOAD(__reg_0, 6);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __STORE(3, __reg_3_0);
    }
    else
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __LOAD(__reg_0, 5);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __LOAD(__reg_0, 6);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __STORE(3, __reg_3_0);
      __DB_SWITCH(); __syncthreads();
    }
    __a_sb = __a_sb_double + __blockSize * 0;
    if (__c1Id == __side1Num - 1)
    {
      for (__h = 7; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - 4;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h - 3, __reg_3_1);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __STORE(__h - 3, __reg_3_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __STORE(__h - 3, __reg_3_0);
        __h++;
        __DB_SWITCH(); __syncthreads();
      }
      if (0) {}
      else if (__h + 1 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_2, __reg_1_2, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h - 3, __reg_3_1);
        __reg_1_1 = __reg_0;
        __CALC2(__reg_2_2, __reg_2_2, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __STORE(__h - 2, __reg_3_2);
        __reg_2_1 = __reg_1_1;
        __CALC3(__reg_3_2, __reg_3_2, __reg_3_0, __reg_2_1);
        __STORE(__h - 1, __reg_3_0);
      }
      else if (__h + 2 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h - 3, __reg_3_1);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_0, __reg_1_0, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __STORE(__h - 2, __reg_3_2);
        __reg_1_2 = __reg_0;
        __CALC2(__reg_2_0, __reg_2_0, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __STORE(__h - 1, __reg_3_0);
        __reg_2_2 = __reg_1_2;
        __CALC3(__reg_3_0, __reg_3_0, __reg_3_1, __reg_2_2);
        __STORE(__h + 0, __reg_3_1);
      }
      else if (__h + 3 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h - 3, __reg_3_1);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __STORE(__h - 2, __reg_3_2);
        __LOAD(__reg_0, __h + 2);
        __CALC1(__reg_1_1, __reg_1_1, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __STORE(__h - 1, __reg_3_0);
        __reg_1_0 = __reg_0;
        __CALC2(__reg_2_1, __reg_2_1, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h + 0, __reg_3_1);
        __reg_2_0 = __reg_1_0;
        __CALC3(__reg_3_1, __reg_3_1, __reg_3_2, __reg_2_0);
        __STORE(__h + 1, __reg_3_2);
      }
    }
    else
    {
      for (__h = 7; __h <= __side1LenOl - 3;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
        __STORE(__h - 3, __reg_3_1);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
        __STORE(__h - 3, __reg_3_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
        __STORE(__h - 3, __reg_3_0);
        __h++;
        __DB_SWITCH(); __syncthreads();
      }
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __CALC3(__reg_3_0, __reg_3_2, __reg_3_1, __reg_2_2);
      __STORE(__h - 3, __reg_3_1);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __CALC3(__reg_3_1, __reg_3_0, __reg_3_2, __reg_2_0);
      __STORE(__h - 3, __reg_3_2);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __CALC3(__reg_3_2, __reg_3_1, __reg_3_0, __reg_2_1);
      __STORE(__h - 3, __reg_3_0);
      __h++;
    }
}
__global__ void kernel0_2(unsigned long long *grid, int x_size_ext, int y_size_ext, int iters, int c0)
{
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
    const AN5D_TYPE __c0Len = (iters - 0);
    const AN5D_TYPE __c0Pad = (0);
    #define __c0 c0
    const AN5D_TYPE __c1Len = (y_size_ext - 1 - 1);
    const AN5D_TYPE __c1Pad = (1);
    #define __c1 c1
    const AN5D_TYPE __c2Len = (x_size_ext - 1 - 1);
    const AN5D_TYPE __c2Pad = (1);
    #define __c2 c2
    const AN5D_TYPE __halo1 = 1;
    const AN5D_TYPE __halo2 = 1;
    const AN5D_TYPE __side0Len = 2;
    const AN5D_TYPE __side1Len = 128;
    const AN5D_TYPE __side2Len = 28;
    const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
    const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
    const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
    const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
    const AN5D_TYPE __blockSize = 1 * __side2LenOl;
    const AN5D_TYPE __side1Num = (__c1Len + __side1Len - 1) / __side1Len;
    const AN5D_TYPE __side2Num = (__c2Len + __side2Len - 1) / __side2Len;
    const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;
    const AN5D_TYPE __local_c2 = __tid;
    const AN5D_TYPE __c1Id = blockIdx.x / __side2Num;
    const AN5D_TYPE __c2 = (blockIdx.x % __side2Num) * __side2Len + __local_c2 + __c2Pad - __OlLen2;
    unsigned long long __reg_0;
    unsigned long long __reg_1_0;
    unsigned long long __reg_1_1;
    unsigned long long __reg_1_2;
    unsigned long long __reg_2_0;
    unsigned long long __reg_2_1;
    unsigned long long __reg_2_2;
    __shared__ unsigned long long __a_sb_double[__blockSize * 2];
    unsigned long long *__a_sb = __a_sb_double;
    const AN5D_TYPE __loadValid = 1 && __c2 >= __c2Pad - __halo2 && __c2 < __c2Pad + __c2Len + __halo2;
    const AN5D_TYPE __updateValid = 1 && __c2 >= __c2Pad && __c2 < __c2Pad + __c2Len;
    const AN5D_TYPE __writeValid1 = __updateValid && __local_c2 >= (__halo2 * 1) && __local_c2 < __side2LenOl - (__halo2 * 1);
    const AN5D_TYPE __writeValid2 = __updateValid && __local_c2 >= (__halo2 * 2) && __local_c2 < __side2LenOl - (__halo2 * 2);
    const AN5D_TYPE __storeValid = __writeValid2;
    AN5D_TYPE __c1;
    AN5D_TYPE __h;
    const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;
    #define __LOAD(reg, h) do { if (__loadValid) { __c1 = __c1Pad2 - __halo1 + h; reg = grid[((__c0 % 2) * y_size_ext + __c1) * x_size_ext + __c2]; }} while (0)
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_0_wrap(__rn0, __a) do { __rn0 = GOL_OP((__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_0(out, a) do { __CALCEXPR_0_wrap(out, a);  } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_1_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_1(out, a) do { unsigned long long etmp; __CALCEXPR_1_wrap(etmp, a); out += etmp; } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_2_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1))); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_2(out, a) do { unsigned long long etmp; __CALCEXPR_2_wrap(etmp, a); out += etmp; } while (0);
    #define __CALCEXPR(out0, out1, out2, reg) do { __CALCEXPR_0(out0, reg); __CALCEXPR_1(out1, reg); __CALCEXPR_2(out2, reg); } while (0);
    #define __CALC1(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid1) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __CALC2(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid2) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __STORE(h, out) do { if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; __DEST = out; }} while (0)
    if (__c1Id == 0)
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __STORE(1, __reg_2_1);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __STORE(2, __reg_2_2);
    }
    else
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __LOAD(__reg_0, 3);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __LOAD(__reg_0, 4);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __STORE(2, __reg_2_2);
      __DB_SWITCH(); __syncthreads();
    }
    __a_sb = __a_sb_double + __blockSize * 1;
    if (__c1Id == __side1Num - 1)
    {
      for (__h = 5; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - 4;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __STORE(__h - 2, __reg_2_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __STORE(__h - 2, __reg_2_1);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __STORE(__h - 2, __reg_2_2);
        __h++;
      }
      if (0) {}
      else if (__h + 1 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_0, __reg_1_0, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __STORE(__h - 2, __reg_2_0);
        __reg_1_2 = __reg_0;
        __CALC2(__reg_2_0, __reg_2_0, __reg_2_1, __reg_1_2);
        __STORE(__h - 1, __reg_2_1);
      }
      else if (__h + 2 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __STORE(__h - 2, __reg_2_0);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_1, __reg_1_1, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __STORE(__h - 1, __reg_2_1);
        __reg_1_0 = __reg_0;
        __CALC2(__reg_2_1, __reg_2_1, __reg_2_2, __reg_1_0);
        __STORE(__h + 0, __reg_2_2);
      }
      else if (__h + 3 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __STORE(__h - 2, __reg_2_0);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __STORE(__h - 1, __reg_2_1);
        __LOAD(__reg_0, __h + 2);
        __CALC1(__reg_1_2, __reg_1_2, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __STORE(__h + 0, __reg_2_2);
        __reg_1_1 = __reg_0;
        __CALC2(__reg_2_2, __reg_2_2, __reg_2_0, __reg_1_1);
        __STORE(__h + 1, __reg_2_0);
      }
    }
    else
    {
      for (__h = 5; __h <= __side1LenOl - 3;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
        __STORE(__h - 2, __reg_2_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
        __STORE(__h - 2, __reg_2_1);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
        __STORE(__h - 2, __reg_2_2);
        __h++;
      }
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __CALC2(__reg_2_2, __reg_2_1, __reg_2_0, __reg_1_1);
      __STORE(__h - 2, __reg_2_0);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __CALC2(__reg_2_0, __reg_2_2, __reg_2_1, __reg_1_2);
      __STORE(__h - 2, __reg_2_1);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __CALC2(__reg_2_1, __reg_2_0, __reg_2_2, __reg_1_0);
      __STORE(__h - 2, __reg_2_2);
      __h++;
    }
}
__global__ void kernel0_1(unsigned long long *grid, int x_size_ext, int y_size_ext, int iters, int c0)
{
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
    const AN5D_TYPE __c0Len = (iters - 0);
    const AN5D_TYPE __c0Pad = (0);
    #define __c0 c0
    const AN5D_TYPE __c1Len = (y_size_ext - 1 - 1);
    const AN5D_TYPE __c1Pad = (1);
    #define __c1 c1
    const AN5D_TYPE __c2Len = (x_size_ext - 1 - 1);
    const AN5D_TYPE __c2Pad = (1);
    #define __c2 c2
    const AN5D_TYPE __halo1 = 1;
    const AN5D_TYPE __halo2 = 1;
    const AN5D_TYPE __side0Len = 1;
    const AN5D_TYPE __side1Len = 128;
    const AN5D_TYPE __side2Len = 30;
    const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
    const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
    const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
    const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
    const AN5D_TYPE __blockSize = 1 * __side2LenOl;
    const AN5D_TYPE __side1Num = (__c1Len + __side1Len - 1) / __side1Len;
    const AN5D_TYPE __side2Num = (__c2Len + __side2Len - 1) / __side2Len;
    const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;
    const AN5D_TYPE __local_c2 = __tid;
    const AN5D_TYPE __c1Id = blockIdx.x / __side2Num;
    const AN5D_TYPE __c2 = (blockIdx.x % __side2Num) * __side2Len + __local_c2 + __c2Pad - __OlLen2;
    unsigned long long __reg_0;
    unsigned long long __reg_1_0;
    unsigned long long __reg_1_1;
    unsigned long long __reg_1_2;
    __shared__ unsigned long long __a_sb_double[__blockSize * 2];
    unsigned long long *__a_sb = __a_sb_double;
    const AN5D_TYPE __loadValid = 1 && __c2 >= __c2Pad - __halo2 && __c2 < __c2Pad + __c2Len + __halo2;
    const AN5D_TYPE __updateValid = 1 && __c2 >= __c2Pad && __c2 < __c2Pad + __c2Len;
    const AN5D_TYPE __writeValid1 = __updateValid && __local_c2 >= (__halo2 * 1) && __local_c2 < __side2LenOl - (__halo2 * 1);
    const AN5D_TYPE __storeValid = __writeValid1;
    AN5D_TYPE __c1;
    AN5D_TYPE __h;
    const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;
    #define __LOAD(reg, h) do { if (__loadValid) { __c1 = __c1Pad2 - __halo1 + h; reg = grid[((__c0 % 2) * y_size_ext + __c1) * x_size_ext + __c2]; }} while (0)
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_0_wrap(__rn0, __a) do { __rn0 = GOL_OP((__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_0(out, a) do { __CALCEXPR_0_wrap(out, a);  } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_1_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1)), (__pet_none), (__pet_none), (__pet_none)); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_1(out, a) do { unsigned long long etmp; __CALCEXPR_1_wrap(etmp, a); out += etmp; } while (0);
    #define __DEST (grid[(((c0 + 1) % 2) * y_size_ext + c1) * x_size_ext + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR_2_wrap(__rn0, __a) do { __rn0 = GOL_OP((__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__pet_none), (__SBREF(__a_sb, -1)), (__REGREF(__a, 0)), (__SBREF(__a_sb, 1))); } while (0)
    #define __DB_SWITCH() do { __a_sb = &__a_sb_double[(__a_sb == __a_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a) do { __DB_SWITCH(); __a_sb[__tid] = a; __syncthreads(); } while (0)
    #define __CALCEXPR_2(out, a) do { unsigned long long etmp; __CALCEXPR_2_wrap(etmp, a); out += etmp; } while (0);
    #define __CALCEXPR(out0, out1, out2, reg) do { __CALCEXPR_0(out0, reg); __CALCEXPR_1(out1, reg); __CALCEXPR_2(out2, reg); } while (0);
    #define __CALC1(out0, out1, out2, reg) do { __CALCSETUP(reg); if (__writeValid1) { __CALCEXPR(out0, out1, out2, reg); } else out1 = reg; } while (0)
    #define __STORE(h, out) do { if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; __DEST = out; }} while (0)
    if (__c1Id == 0)
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __STORE(1, __reg_1_1);
    }
    else
    {
      __LOAD(__reg_0, 0);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __LOAD(__reg_0, 1);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __LOAD(__reg_0, 2);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __STORE(1, __reg_1_1);
    }
    __a_sb = __a_sb_double + __blockSize * 1;
    if (__c1Id == __side1Num - 1)
    {
      for (__h = 3; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - 4;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __STORE(__h - 1, __reg_1_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __STORE(__h - 1, __reg_1_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __STORE(__h - 1, __reg_1_1);
        __h++;
        __DB_SWITCH(); __syncthreads();
      }
      if (0) {}
      else if (__h + 1 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_1, __reg_1_2, __reg_0);
        __STORE(__h - 1, __reg_1_2);
      }
      else if (__h + 2 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __STORE(__h - 1, __reg_1_2);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_2, __reg_1_2, __reg_1_0, __reg_0);
        __STORE(__h + 0, __reg_1_0);
      }
      else if (__h + 3 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0, __h + 0);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __STORE(__h - 1, __reg_1_2);
        __LOAD(__reg_0, __h + 1);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __STORE(__h + 0, __reg_1_0);
        __LOAD(__reg_0, __h + 2);
        __CALC1(__reg_1_0, __reg_1_0, __reg_1_1, __reg_0);
        __STORE(__h + 1, __reg_1_1);
      }
    }
    else
    {
      for (__h = 3; __h <= __side1LenOl - 3;)
      {
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
        __STORE(__h - 1, __reg_1_2);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
        __STORE(__h - 1, __reg_1_0);
        __h++;
        __LOAD(__reg_0, __h);
        __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
        __STORE(__h - 1, __reg_1_1);
        __h++;
        __DB_SWITCH(); __syncthreads();
      }
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_1, __reg_1_0, __reg_1_2, __reg_0);
      __STORE(__h - 1, __reg_1_2);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_2, __reg_1_1, __reg_1_0, __reg_0);
      __STORE(__h - 1, __reg_1_0);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0, __h);
      __CALC1(__reg_1_0, __reg_1_2, __reg_1_1, __reg_0);
      __STORE(__h - 1, __reg_1_1);
      __h++;
    }
}
