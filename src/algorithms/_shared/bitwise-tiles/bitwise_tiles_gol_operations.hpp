#ifndef ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP
#define ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP

#include <bitset>
#include <cstddef>
#include <iostream>
#include <sstream>

namespace algorithms {

struct BitwiseTileOps {
    using tile_type = std::uint64_t;

    template <tile_type... rows>
    static constexpr tile_type tile_num() {
        tile_type result = 0;
        ((result = (result << 8) | rows), ...);
        return result;
    }

    // clang-format off
    static tile_type compute_center_tile(
        tile_type lt, tile_type ct, tile_type rt, 
        tile_type lc, tile_type cc, tile_type rc,
        tile_type lb, tile_type cb, tile_type rb) {
        
        tile_type result = compute_inner_bits(cc);

        result |= compute_side<Vertical::TOP>(cc, ct);
        result |= compute_side<Vertical::BOTTOM>(cc, cb);
        result |= compute_side<Horizontal::LEFT>(cc, lc);
        result |= compute_side<Horizontal::RIGHT>(cc, rc);

        result |= compute_corner_top_left(lt, ct, 
                                          lc, cc);
        result |= compute_corner_top_right(ct, rt, 
                                           cc, rc);
        result |= compute_corner_bottom_left(lc, cc, 
                                             lb, cb);
        result |= compute_corner_bottom_right(cc, rc, 
                                              cb, rb);
        return result;
    }
    // clang-format on

    static std::string debug_print(tile_type tile) {
        std::ostringstream oss;

        for (int i = 7; i >= 0; --i) {
            std::ostringstream line;
            line << std::bitset<8>(tile >> (8 * i));

            for (char ch : line.str()) {
                if (ch == '0') {
                    oss << "\033[30m" << ch << "\033[0m";
                }
                else {
                    oss << "\033[31m" << ch << "\033[0m";
                }
            }

            oss << std::endl;
        }

        return oss.str();
    }
    
public: // Ensure enums are public
    enum class Horizontal : char {
        LEFT = 0,
        RIGHT = 1,
    };

    enum class Vertical : char {
        TOP = 0,
        BOTTOM = 1,
    };

private:

    static tile_type compute_inner_bits(tile_type tile) {
        tile_type result = 0;
        
        constexpr tile_type neighborhood_mask = tile_num<
            0b0000'0111,
            0b0000'0101,
            0b0000'0111>();

        constexpr tile_type cell_mask = tile_num<
            0b0000'0000,
            0b0000'0010,
            0b0000'0000>();

        constexpr tile_type one = tile_num<
            0b1000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000,
            0b0000'0000>();

        for (std::size_t row = 0; row < 6; ++row) {
            for (std::size_t col = 0; col < 6; ++col) {

                auto neighborhood = tile & neighborhood_mask;
                auto cell = tile & cell_mask;

                auto alive_neighbours = __builtin_popcount(neighborhood);

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

                tile >>= 1;
                result >>= 1;
            }

            tile >>= 2;
            result >>= 2;
        }

        result >>= 6;

        return result;
    }

    template <Vertical Site>
    static tile_type compute_side(tile_type tile, tile_type site_tile) {
        tile_type result = 0;
        tile_type site_tile_mask;
        tile_type center_tile_mask;
        tile_type cell;

        if constexpr (Site == Vertical::TOP) {
            site_tile_mask = 0b1110'0000;
            center_tile_mask = tile_num<
                0b1010'0000,
                0b1110'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000
            >();
            cell = tile_num<
                0b0100'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000
            >();
        } else {
            site_tile_mask = tile_num<
                0b1110'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000,
                0b0000'0000
            >(); 
            center_tile_mask = tile_num<
                0b1010'0000,
                0b1110'0000
            >();
            cell = 0b0100'0000;
        }

        for (size_t i = 0; i < 6; ++i) {
            auto site_neighborhood = site_tile & site_tile_mask;
            auto center_neighborhood = tile & center_tile_mask;

            auto alive_neighbours = __builtin_popcount(site_neighborhood) +
                __builtin_popcount(center_neighborhood);

            if (center_neighborhood != 0) {
                if (alive_neighbours < 2 || alive_neighbours > 3) {
                    result &= ~cell;
                } 
                else {
                    result |= cell;
                }
            } 
            else {
                if (alive_neighbours == 3) {
                    result |= cell;
                } 
                else {
                    result &= ~cell;
                }
            }

            site_tile >>= 1;
            tile >>= 1;
            result >>= 1;
        }
        
        result >>= 1;

        if constexpr (Site == Vertical::BOTTOM) {
            result >>= 56;
        }

        return result;
    }
    
    template <Horizontal Site>
    static tile_type compute_side(tile_type tile, tile_type site_tile) {
        tile_type result = 0;
        return result;
    }

    template <Horizontal HorizontalSite, Vertical VerticalSite>
    static tile_type compute_corner(tile_type t1, tile_type t2,
                                    tile_type t3, tile_type t4);

    static tile_type compute_corner_top_left(tile_type lt, tile_type ct,
                                             tile_type lc, tile_type cc) {
        
        tile_type result = 0;
        return result;
    }

    static tile_type compute_corner_top_right(tile_type ct, tile_type rt,
                                              tile_type cc, tile_type rc) {
        
        tile_type result = 0;
        return result;
    }

    static tile_type compute_corner_bottom_left(tile_type lc, tile_type cc,
                                                tile_type lb, tile_type cb) {
        
        tile_type result = 0;
        return result;
    }

    static tile_type compute_corner_bottom_right(tile_type cc, tile_type rc,
                                                 tile_type cb, tile_type rb) {
        
        tile_type result = 0;
        return result;
    }
};

} // namespace algorithms

#endif // ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP