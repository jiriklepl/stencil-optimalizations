#ifndef GOL_BIT_GRID_HPP
#define GOL_BIT_GRID_HPP

#include "../../infrastructure/grid.hpp"
#include <cstddef>

namespace algorithms {

class GolBitGrid {
  public:
    using tile_type = uint64_t;
    using size_type = std::size_t;
    using Grid = infrastructure::Grid<2, char>;

    GolBitGrid(const Grid& grid) {
        assert_dim_has_correct_size<0>(grid);
        assert_dim_has_correct_size<1>(grid);

        x_tiles_count = grid.size_in<0>() / 8;
        y_tiles_count = grid.size_in<1>() / 8;

        tiles.resize((x_tiles_count + 2) * (y_tiles_count + 2), 0);
    };

    tile_type get_tile(int x, int y) {
        auto x_with_offset = x + 1;
        auto y_with_offset = y + 1;

        return tiles[y_with_offset * (x_tiles_count + 2) + x_with_offset];
    }

    void set_tile(int x, int y, tile_type tile) {
        auto x_with_offset = x + 1;
        auto y_with_offset = y + 1;

        tiles[y_with_offset * (x_tiles_count + 2) + x_with_offset] = tile;
    }

  private:
    size_type x_tiles_count;
    size_type y_tiles_count;
    std::vector<tile_type> tiles;

    template <int DIM>
    void assert_dim_has_correct_size(const Grid& grid) {
        if (grid.size_in<DIM>() % 8 != 0) {
            throw std::invalid_argument("Grid dimensions must be a multiple of 8");
        }
    }
};

} // namespace algorithms

#endif // GOL_BIT_GRID_HPP