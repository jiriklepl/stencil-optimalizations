#ifndef GOL_CPU_NAIVE_HPP
#define GOL_CPU_NAIVE_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-tiles/bit_tiles_grid.hpp"
#include "../_shared/bitwise-tiles/bitwise_tiles_gol_operations.hpp"

namespace algorithms {

class GoLCpuBitwise : public infrastructure::Algorithm<2, char> {
  public:
    using DataGrid = infrastructure::Grid<2, char>;
    using tile_type = GolBitTilesGrid::tile_type;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = GolBitTilesGrid(data);
    }

    void initialize_data_structures() override {
    }

    void run(size_type iterations) override {
        auto x_size = bit_grid.x_tiles();
        auto y_size = bit_grid.y_tiles();

        for (size_type i = 0; i < iterations; ++i) {
            for (size_type x = 0; x < x_size; ++x) {
                for (size_type y = 0; y < y_size; ++y) {
                    // clang-format off

                    tile_type lt, ct, rt;
                    tile_type lc, cc, rc; 
                    tile_type lb, cb, rb;

                    load(bit_grid, x, y,
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb);

                    tile_type new_center = BitwiseTileOps::compute_center_tile(
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb
                    );

                    bit_grid.set_tile(x, y, new_center);

                    // clang-format on
                }
            }
        }
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return _result;
    }

  private:
    void load(const GolBitTilesGrid& bit_grid, size_type x, size_type y, tile_type& lt, tile_type& ct, tile_type& rt,
              tile_type& lc, tile_type& cc, tile_type& rc, tile_type& lb, tile_type& cb, tile_type& rb) {
        lt = bit_grid.get_tile(x - 1, y - 1);
        ct = bit_grid.get_tile(x, y - 1);
        rt = bit_grid.get_tile(x + 1, y - 1);

        lc = bit_grid.get_tile(x - 1, y);
        cc = bit_grid.get_tile(x, y);
        rc = bit_grid.get_tile(x + 1, y);

        lb = bit_grid.get_tile(x - 1, y + 1);
        cb = bit_grid.get_tile(x, y + 1);
        rb = bit_grid.get_tile(x + 1, y + 1);
    }

    DataGrid _result;
    GolBitTilesGrid bit_grid;
};

} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP