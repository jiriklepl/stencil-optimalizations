#include "cpu_gol.hpp"
#include "an5d-cpu-alg.hpp"

namespace algorithms {

template <>
void An5dAlg<32, ExecModel::CPU>::run(An5dAlg<32, ExecModel::CPU>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(), iterations);
}

template <>
void An5dAlg<64, ExecModel::CPU>::run(An5dAlg<64, ExecModel::CPU>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(), iterations);
}

#undef GOL_OP

}

