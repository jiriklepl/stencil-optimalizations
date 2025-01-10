#ifndef CUDA_NAIVE_LOCAL_MODELS_HPP
#define CUDA_NAIVE_LOCAL_MODELS_HPP

#include <cstddef>
namespace algorithms::cuda_naive_local {

struct Dims {
    std::size_t x;
    std::size_t y;
};

template <typename change_state_store_type>
struct ChangeStateStore {
    change_state_store_type* last;
    change_state_store_type* current;
};

template <typename col_type, typename change_state_store_type>
struct BitGridWithChangeInfo {
    constexpr static std::size_t BITS = sizeof(col_type) * 8;

    col_type* input;
    col_type* output;

    std::size_t x_size;
    std::size_t y_size;

    Dims warp_dims;
    Dims warp_tile_dims;

    ChangeStateStore<change_state_store_type> change_state_store;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP