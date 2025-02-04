#ifndef ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH
#define ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH

#include <cstdint>
#include "./macro-cols.hpp"
#include "./macro-tiles.hpp"
#include <cuda_runtime.h> 
#include "../bit_modes.hpp"
#include "./wasteful-rows.cuh"

namespace algorithms {

#undef POPCOUNT_16
#undef POPCOUNT_32
#undef POPCOUNT_64

#define POPCOUNT_16(x) __popc(x)
#define POPCOUNT_32(x) __popc(x)
#define POPCOUNT_64(x) __popcll(x)

template <typename word_type, typename bit_grid_model>
class CudaBitwiseOps {};

template <>
class CudaBitwiseOps<std::uint16_t, BitColumnsMode> {
    using word_type = std::uint16_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __16_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};


template <>
class CudaBitwiseOps<std::uint32_t, BitColumnsMode> {
    using word_type = std::uint32_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t, BitColumnsMode> {
    using word_type = std::uint64_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t, BitColumnsFujitaMode> {
    using word_type = std::uint64_t;

    __device__ static constexpr __forceinline__ word_type compute_center_word_fujita(
        word_type A, word_type B, word_type C, 
        word_type H, word_type I, word_type D,
        word_type G, word_type F, word_type E) {
        // 1.
        const word_type AB_1 = A & B;
        const word_type AB_0 = A ^ B;
        // 2.
        const word_type CD_1 = C & D;
        const word_type CD_0 = C ^ D;
        // 3.
        const word_type EF_1 = E & F;
        const word_type EF_0 = E ^ F;
        // 4.
        const word_type GH_1 = G & H;
        const word_type GH_0 = G ^ H;
        // 5.
        const word_type AD_0 = AB_0 ^ CD_0;
        // 6.
        const word_type AD_1 = AB_1 ^ CD_1 ^ (AB_0 & CD_0);
        // 7.
        const word_type AD_2 = AB_1 & CD_1;
        // 8.
        const word_type EH_0 = EF_0 ^ GH_0;
        // 9.
        const word_type EH_1 = EF_1 ^ GH_1 ^ (EF_0 & GH_0);
        // 10.
        const word_type EH_2 = EF_1 & GH_1;
        // 11.
        const word_type AH_0 = AD_0 ^ EH_0;
        // 12.
        const word_type X = AD_0 & EH_0;
        // 13.
        const word_type Y = AD_1 ^ EH_1;
        // 14.
        const word_type AH_1 = X ^ Y;
        // 15.
        const word_type AH_23 = AD_2 | EH_2 | (AD_1 & EH_1) | (X & Y);
        // 17. neither of the 2 most significant bits is set and the second least significant bit is set
        const word_type Z = ~AH_23 & AH_1;
        // 18. (two neighbors) the least significant bit is not set and Z
        const word_type I_2 = ~AH_0 & Z;
        // 19. (three neighbors) the least significant bit is set and Z
        const word_type I_3 = AH_0 & Z;
        // 20.
        return (I & I_2) | I_3;
    }

public:
    __device__ static constexpr __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
        // the top-left neighbors of the center cell:
        const word_type A = (lt << 63) | (ct >> 1);
        // the top neighbors of the center cell:
        const word_type B = ct;
        // the top-right neighbors of the center cell:
        const word_type C = (ct << 1) | (rt >> 63);
        // the right neighbors of the center cell:
        const word_type D = (cc << 1) | (rc >> 63);
        // the bottom-right neighbors of the center cell:
        const word_type E = (cb << 1) | (rb >> 63);
        // the bottom neighbors of the center cell:
        const word_type F = cb;
        // the bottom-left neighbors of the center cell:
        const word_type G = (cb >> 1) | (lb << 63);
        // the left neighbors of the center cell:
        const word_type H = (cc >> 1) | (lc << 63);
        const word_type I = cc;

        return compute_center_word_fujita(A, B, C, H, I, D, G, F, E);
    }
};

template <>
class CudaBitwiseOps<std::uint16_t, BitTileMode> {
    using word_type = std::uint16_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __16_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};


template <>
class CudaBitwiseOps<std::uint32_t, BitTileMode> {
    using word_type = std::uint32_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __32_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t, BitTileMode> {
    using word_type = std::uint64_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, BitWastefulRowsMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return WastefulRowsImplantation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

} // namespace algorithms
#endif // ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH