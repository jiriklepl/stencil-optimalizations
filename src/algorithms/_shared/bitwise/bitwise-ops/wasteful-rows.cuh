#ifndef ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_WASTEFUL_ROWS_CUH
#define ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_WASTEFUL_ROWS_CUH
    

#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>

namespace algorithms {

template <typename word_type>
struct WastefulRowsImplantation {

    constexpr static std::size_t BITS_PER_CELL = WastefulRows<word_type>::BITS_PER_CELL;
    constexpr static std::size_t BITS = WastefulRows<word_type>::BITS;
    constexpr static std::size_t CELLS_PER_WORD = WastefulRows<word_type>::CELLS_PER_WORD;

    constexpr static word_type CELL_MASK = (static_cast<word_type>(1) << BITS_PER_CELL) - 1;

    static __host__ __device__ __forceinline__  word_type compute_center_word_simple(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type neighborhoods = ct + cc + cb;
        neighborhoods += (neighborhoods >> BITS_PER_CELL) + (neighborhoods << BITS_PER_CELL);

        word_type right_neighborhoods = (rt + rc + rb) << (BITS - BITS_PER_CELL);
        word_type left_neighborhoods = (lt + lc + lb) >> (BITS - BITS_PER_CELL);

        neighborhoods += right_neighborhoods + left_neighborhoods;

        neighborhoods -= cc;

        word_type result = 0;

        for (std::size_t i = 0; i < CELLS_PER_WORD; i++) {

            word_type neighbors_count = ((neighborhoods >> (i * BITS_PER_CELL)) & CELL_MASK);
            word_type cell_alive = (cc >> (i * BITS_PER_CELL)) & 1;

            if (cell_alive) {
                if (neighbors_count == 2 || neighbors_count == 3) {
                    result |= (static_cast<word_type>(1) << (i * BITS_PER_CELL));
                }
            } else {
                if (neighbors_count == 3) {
                    result |= (static_cast<word_type>(1) << (i * BITS_PER_CELL));
                }
            }
        }        

        return result;
    }

    constexpr static word_type bit_table = static_cast<word_type>(0b0000'0110'0000'1000);

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type neighborhoods = ct + cc + cb;
        neighborhoods += (neighborhoods >> BITS_PER_CELL) + (neighborhoods << BITS_PER_CELL);

        word_type right_neighborhoods = (rt + rc + rb) << (BITS - BITS_PER_CELL);
        word_type left_neighborhoods = (lt + lc + lb) >> (BITS - BITS_PER_CELL);

        neighborhoods += right_neighborhoods + left_neighborhoods;
        neighborhoods += (cc << 2) + (cc << 1);

        word_type result = 0;

        for (std::size_t i = 0; i < CELLS_PER_WORD; i++) {
            word_type idx = neighborhoods & CELL_MASK;
            
            word_type cell_res = (bit_table >> idx) & 1;
            result |= (cell_res << (i * BITS_PER_CELL));

            neighborhoods >>= BITS_PER_CELL;
        }        

        return result;
    }
};

}

#endif // ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_WASTEFUL_ROWS_CUH