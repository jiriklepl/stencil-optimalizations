// ...existing code...
#include "../template_helpers/static_for.hpp"
#include "bitwise_cols_gol_operations.hpp"
#include <cstdint>

namespace algorithms {

// clang-format off
template <typename col_type>
inline col_type BitwiseColsOps<col_type>::compute_center_col(
    col_type lt, col_type ct, col_type rt, 
    col_type lc, col_type cc, col_type rc,
    col_type lb, col_type cb, col_type rb) {

    col_type result = compute_inner_bits(lc, cc, rc);

    result |= compute_side_col<Position::TOP>(
        lc, cc, rc,
        lt, ct, rt);

    result |= compute_side_col<Position::BOTTOM>(
        lc, cc, rc,
        lb, cb, rb);

    return result;
}
// clang-format on

template <typename col_type>
inline col_type BitwiseColsOps<col_type>::compute_inner_bits(col_type lc, col_type cc, col_type rc) {
    col_type result = 0;

    templates::static_for<1, BITS_IN_COL - 1>::run(
        [&lc, &cc, &rc, &result]<std::size_t N>() { result |= compute_inner_cell<N>(lc, cc, rc); });

    return result;
}

template <typename col_type>
template <std::size_t N>
inline col_type BitwiseColsOps<col_type>::compute_inner_cell(col_type lc, col_type cc, col_type rc) {
    col_type result = 0;
    constexpr col_type cell_mask = static_cast<col_type>(0b010) << (N - 1);
    constexpr col_type one = cell_mask;

    auto cell = cc & cell_mask;

    auto neighborhood = combine_neighborhoods_into_one_word<N>(lc, cc, rc);
    auto alive_neighbours = __builtin_popcountll(neighborhood);

    // auto alive_neighbours =
    //     __builtin_popcountll(lc & cell_mask) +
    //     __builtin_popcountll(cc & cell_mask) +
    //     __builtin_popcountll(rc & cell_mask);

    if (cell != 0) {
        if (alive_neighbours < 2 || alive_neighbours > 3) {
            result &= ~one;
        }
        else {
            result |= one;
        }
    }
    else {
        if (alive_neighbours == 3) {
            result |= one;
        }
        else {
            result &= ~one;
        }
    }

    return result;
}

template <typename col_type>
template <std::size_t N>
inline col_type BitwiseColsOps<col_type>::combine_neighborhoods_into_one_word(col_type lc, col_type cc, col_type rc) {
    constexpr col_type site_neighborhood_mask = static_cast<col_type>(0b111) << (N - 1);
    constexpr col_type center_neighborhood_mask = static_cast<col_type>(0b101) << (N - 1);
    constexpr col_type NEIGHBORHOOD_WINDOW = 6;

    return offset<6, N - 1, NEIGHBORHOOD_WINDOW>(lc & site_neighborhood_mask) |
           offset<3, N - 1, NEIGHBORHOOD_WINDOW>(cc & center_neighborhood_mask) | (rc & site_neighborhood_mask);
}

template <typename col_type>
template <std::size_t N, std::size_t CENTER, std::size_t NEIGHBORHOOD_WINDOW>
inline col_type BitwiseColsOps<col_type>::offset(col_type num) {
    if constexpr (CENTER < NEIGHBORHOOD_WINDOW) {
        return num << N;
    }
    else {
        return num >> N;
    }
}

// clang-format off
template <typename col_type>
template <Position POSITION>
inline col_type BitwiseColsOps<col_type>::compute_side_col(
    col_type cl, col_type cc, col_type cr,
    col_type l_, col_type c_, col_type r_
) {
    constexpr col_type SITE_MASK = masks<POSITION>::SITE_MASK;
    constexpr col_type CENTER_MASK = masks<POSITION>::CENTER_MASK;
    constexpr col_type UP_BOTTOM_MASK = masks<POSITION>::UP_BOTTOM_MASK;
    constexpr col_type CELL_MASK = masks<POSITION>::CELL_MASK;

    auto neighborhood = 
        masks<POSITION>::template offset_center_cols<7>(cl & SITE_MASK) | 
        masks<POSITION>::template offset_center_cols<5>(cc & CENTER_MASK) |
        masks<POSITION>::template offset_center_cols<3>(cr & SITE_MASK) |
        masks<POSITION>::template offset_top_bottom_cols<2>(l_ & UP_BOTTOM_MASK) |
        masks<POSITION>::template offset_top_bottom_cols<1>(c_ & UP_BOTTOM_MASK) |
                                                           (r_ & UP_BOTTOM_MASK);
                                                           
    auto alive_neighbours = __builtin_popcountll(neighborhood);

    auto cell = cc & CELL_MASK;

    // auto alive_neighbours = 
    //     __builtin_popcountll(cl & SITE_MASK) +
    //     __builtin_popcountll(cc & CENTER_MASK) +
    //     __builtin_popcountll(cr & SITE_MASK) +
    //     __builtin_popcountll(l_ & UP_BOTTOM_MASK) +
    //     __builtin_popcountll(c_ & UP_BOTTOM_MASK) +
    //     __builtin_popcountll(r_ & UP_BOTTOM_MASK);


    if (cell != 0) {
        if (alive_neighbours < 2 || alive_neighbours > 3) {
            return 0;
        }
        else {
            return CELL_MASK;
        }
    }
    else {
        if (alive_neighbours == 3) {
            return CELL_MASK;
        }
        else {
            return 0;
        }
    }
}
// clang-format on

template <typename col_type>
template <std::size_t N>
inline col_type MasksByPosition<Position::TOP, col_type>::offset_center_cols(col_type num) {
    return num << N;
}

template <typename col_type>
template <std::size_t N>
inline col_type MasksByPosition<Position::TOP, col_type>::offset_top_bottom_cols(col_type num) {
    return num >> N;
}

template <typename col_type>
template <std::size_t N>
inline col_type MasksByPosition<Position::BOTTOM, col_type>::offset_center_cols(col_type num) {
    return num >> N;
}

template <typename col_type>
template <std::size_t N>
inline col_type MasksByPosition<Position::BOTTOM, col_type>::offset_top_bottom_cols(col_type num) {
    return num << N;
}

} // namespace algorithms

template class algorithms::BitwiseColsOps<std::uint8_t>;
template class algorithms::BitwiseColsOps<std::uint16_t>;
template class algorithms::BitwiseColsOps<std::uint32_t>;
template class algorithms::BitwiseColsOps<std::uint64_t>;

template class algorithms::MasksByPosition<algorithms::Position::TOP, std::uint8_t>;
template class algorithms::MasksByPosition<algorithms::Position::TOP, std::uint16_t>;
template class algorithms::MasksByPosition<algorithms::Position::TOP, std::uint32_t>;
template class algorithms::MasksByPosition<algorithms::Position::TOP, std::uint64_t>;
template class algorithms::MasksByPosition<algorithms::Position::BOTTOM, std::uint8_t>;
template class algorithms::MasksByPosition<algorithms::Position::BOTTOM, std::uint16_t>;
template class algorithms::MasksByPosition<algorithms::Position::BOTTOM, std::uint32_t>;
template class algorithms::MasksByPosition<algorithms::Position::BOTTOM, std::uint64_t>;
