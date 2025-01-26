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
    change_state_store_type* before_last;
    change_state_store_type* last;
    change_state_store_type* current;
};


template <typename word_type, typename change_state_store_type>
struct BitGridWithChangeInfo {
    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    word_type* input;
    word_type* output;

    std::size_t x_size;
    std::size_t y_size;

    Dims warp_dims;
    Dims warp_tile_dims;
    Dims block_dims;

    ChangeStateStore<change_state_store_type> change_state_store;
};

template <typename word_type>
struct BitGridWithTiling {
    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    word_type* input;
    word_type* output;

    std::size_t x_size;
    std::size_t y_size;

    Dims warp_dims;
    Dims warp_tile_dims;
    Dims block_dims;
};

template <typename word_type, typename idx_t>
struct WarpInformation {
    idx_t warp_idx;
    idx_t lane_idx;

    idx_t x_block_abs_size;
    idx_t y_block_abs_size;

    idx_t x_block_count;

    idx_t x_block;
    idx_t y_block;

    idx_t x_warp;
    idx_t y_warp;

    idx_t x_abs_start;
    idx_t y_abs_start;

    idx_t x_computed_word_count;
    idx_t y_computed_word_count;
};

template <typename state_store_type, typename idx_t>
struct StateStoreInfo {
    state_store_type* cached_state;
    static constexpr std::size_t CACHE_SIZE_X = 3;
    static constexpr std::size_t CACHE_SIZE_Y = 3;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP