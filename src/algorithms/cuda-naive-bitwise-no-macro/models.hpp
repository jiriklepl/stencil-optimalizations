#ifndef CUDA_NAIVE_MODELS_BIT_GRID_NO_MACRO_HPP
#define CUDA_NAIVE_MODELS_BIT_GRID_NO_MACRO_HPP

#include <cstddef>
namespace algorithms {

template <typename col_type, typename idx_t>
struct BitGridOnCudaWitOriginalSizes {
    col_type* input;
    col_type* output;
    
    idx_t x_size;
    idx_t y_size;

    idx_t x_size_original;
    idx_t y_size_original;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP