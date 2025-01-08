#ifndef GOL_OPS_FOR_AN5D_HPP
#define GOL_OPS_FOR_AN5D_HPP

#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"

#ifdef AN5D_TYPE
    // code for AN5D generated code
    #define GOL_OP(lt, tt, rt, lc, cc, rc, lb, cb, rb) \
        lt + tt + rt + lc + rc + lb + cb + rb + cc
#else
    // code for the raw C code on CPU
    
    #include <cstdint>
    
    #define POPCOUNT_16(x) __builtin_popcount(x)
    #define POPCOUNT_32(x) __builtin_popcount(x)
    #define POPCOUNT_64(x) __builtin_popcountll(x)
    
    #define GOL_OP(lt, ct, rt, lc, cc, rc, lb, cb, rb) __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)

#endif

#endif // GOL_OPS_FOR_AN5D_HPP