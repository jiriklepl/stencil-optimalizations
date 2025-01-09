#include "an5d-cpu-alg.hpp"
#include "cpu_gol.hpp"
#include "cuda_gol_32_64_an5d.hpp"

namespace algorithms {

template <>
void An5dAlg<32, ExecModel::CPU>::run(An5dAlg<32, ExecModel::CPU>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<64, ExecModel::CPU>::run(An5dAlg<64, ExecModel::CPU>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<32, ExecModel::CUDA>::run(An5dAlg<32, ExecModel::CUDA>::size_type iterations) {
    cuda_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<64, ExecModel::CUDA>::run(An5dAlg<64, ExecModel::CUDA>::size_type iterations) {
    cuda_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

#undef GOL_OP

} // namespace algorithms
