#include "raw_gol.hpp"
#include <iostream>
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "an5d-cpu-alg.hpp"

namespace algorithms {

template <>
void An5dCpu<32>::run(An5dCpu<32>::size_type iterations) {
    gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(), iterations);
}

#undef GOL_OP
#define GOL_OP(lt, ct, rt, lc, cc, rc, lb, cb, rb) __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)

// template <>
// void An5dCpu<64>::run(An5dCpu<32>::size_type iterations) {
//     gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(), iterations);
// }

#undef GOL_OP

}

