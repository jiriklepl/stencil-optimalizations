#ifndef ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP
#define ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP

#include "../template_helpers/static_for.hpp"
#include <bitset>
#include <cstddef>
#include <iostream>
#include <sstream>

namespace algorithms {

enum class Position {
    TOP = 0,
    BOTTOM = 1,
};

template <Position POSITION, typename col_type>
class MasksByPosition {};

template <typename col_type>
struct BitwiseColsOps {
    constexpr static std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    template <Position POSITION>
    using masks = MasksByPosition<POSITION, col_type>;

    // clang-format off
    static  col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb);
    // clang-format on

  private:
    static col_type compute_inner_bits(col_type lc, col_type cc, col_type rc);

    template <std::size_t N>
    static col_type compute_inner_cell(col_type lc, col_type cc, col_type rc);

    template <std::size_t N>
    static col_type combine_neighborhoods_into_one_word(col_type lc, col_type cc, col_type rc);

    template <std::size_t N, std::size_t CENTER, std::size_t NEIGHBORHOOD_WINDOW>
    static col_type offset(col_type num);

    // clang-format off
    template <Position POSITION>
    static col_type compute_side_col(
        col_type cl, col_type cc, col_type cr,
        col_type l_, col_type c_, col_type r_);
    // clang-format on
};

template <typename col_type>
class MasksByPosition<Position::TOP, col_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    static constexpr col_type SITE_MASK = 0b11;
    static constexpr col_type CENTER_MASK = 0b10;
    static constexpr col_type UP_BOTTOM_MASK = static_cast<col_type>(1) << (BITS_IN_COL - 1);
    static constexpr col_type CELL_MASK = 0b1;

    template <std::size_t N>
    static col_type offset_center_cols(col_type num);

    template <std::size_t N>
    static col_type offset_top_bottom_cols(col_type num);
};

template <typename col_type>
class MasksByPosition<Position::BOTTOM, col_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    static constexpr col_type SITE_MASK = static_cast<col_type>(0b11) << (BITS_IN_COL - 2);
    static constexpr col_type CENTER_MASK = static_cast<col_type>(0b01) << (BITS_IN_COL - 2);
    static constexpr col_type UP_BOTTOM_MASK = 1;
    static constexpr col_type CELL_MASK = static_cast<col_type>(0b1) << (BITS_IN_COL - 1);

    template <std::size_t N>
    static col_type offset_center_cols(col_type num);

    template <std::size_t N>
    static col_type offset_top_bottom_cols(col_type num);
};

} // namespace algorithms

#endif // ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP