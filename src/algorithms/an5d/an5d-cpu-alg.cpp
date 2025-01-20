#include "an5d-cpu-alg.hpp"
#include "cpu_gol.hpp"
#include "cuda_gol_32_64_an5d.hpp"
#include "../_shared/common_grid_types.hpp"

namespace algorithms {

template <>
void An5dAlg<common::CHAR, 32, ExecModel::CPU, BitColumnsMode>::run(An5dAlg<common::CHAR, 32, ExecModel::CPU, BitColumnsMode>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::CHAR, 64, ExecModel::CPU, BitColumnsMode>::run(An5dAlg<common::CHAR, 64, ExecModel::CPU, BitColumnsMode>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::CHAR, 32, ExecModel::CUDA, BitColumnsMode>::run(An5dAlg<common::CHAR, 32, ExecModel::CUDA, BitColumnsMode>::size_type iterations) {
    cuda_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::CHAR, 64, ExecModel::CUDA, BitColumnsMode>::run(An5dAlg<common::CHAR, 64, ExecModel::CUDA, BitColumnsMode>::size_type iterations) {
    cuda_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::INT, 32, ExecModel::CPU, BitColumnsMode>::run(An5dAlg<common::INT, 32, ExecModel::CPU, BitColumnsMode>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::INT, 64, ExecModel::CPU, BitColumnsMode>::run(An5dAlg<common::INT, 64, ExecModel::CPU, BitColumnsMode>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::INT, 32, ExecModel::CUDA, BitColumnsMode>::run(An5dAlg<common::INT, 32, ExecModel::CUDA, BitColumnsMode>::size_type iterations) {
    cuda_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::INT, 64, ExecModel::CUDA, BitColumnsMode>::run(An5dAlg<common::INT, 64, ExecModel::CUDA, BitColumnsMode>::size_type iterations) {
    cuda_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}


template <>
void An5dAlg<common::CHAR, 32, ExecModel::CPU, BitTileMode>::run(An5dAlg<common::CHAR, 32, ExecModel::CPU, BitTileMode>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::CHAR, 64, ExecModel::CPU, BitTileMode>::run(An5dAlg<common::CHAR, 64, ExecModel::CPU, BitTileMode>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::CHAR, 32, ExecModel::CUDA, BitTileMode>::run(An5dAlg<common::CHAR, 32, ExecModel::CUDA, BitTileMode>::size_type iterations) {
    cuda_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::CHAR, 64, ExecModel::CUDA, BitTileMode>::run(An5dAlg<common::CHAR, 64, ExecModel::CUDA, BitTileMode>::size_type iterations) {
    cuda_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::INT, 32, ExecModel::CPU, BitTileMode>::run(An5dAlg<common::INT, 32, ExecModel::CPU, BitTileMode>::size_type iterations) {
    cpu_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::INT, 64, ExecModel::CPU, BitTileMode>::run(An5dAlg<common::INT, 64, ExecModel::CPU, BitTileMode>::size_type iterations) {
    cpu_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
               iterations);
}

template <>
void An5dAlg<common::INT, 32, ExecModel::CUDA, BitTileMode>::run(An5dAlg<common::INT, 32, ExecModel::CUDA, BitTileMode>::size_type iterations) {
    cuda_gol_32(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}

template <>
void An5dAlg<common::INT, 64, ExecModel::CUDA, BitTileMode>::run(An5dAlg<common::INT, 64, ExecModel::CUDA, BitTileMode>::size_type iterations) {
    cuda_gol_64(input_bit_grid->data(), result_bit_grid->data(), input_bit_grid->x_size(), input_bit_grid->y_size(),
                iterations);
}


#undef GOL_OP

} // namespace algorithms
