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
};

template <typename word_type, typename idx_t>
struct WarpInformation {
    idx_t warp_idx;
    idx_t lane_idx;

    idx_t x_tile;
    idx_t y_tile;

    idx_t x_tiles;
    idx_t y_tiles;

    idx_t x_in_warp;
    idx_t y_in_warp;

    idx_t x_cols_in_warp;
    idx_t y_rows_in_warp;

    idx_t x_start;
    idx_t y_start;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP