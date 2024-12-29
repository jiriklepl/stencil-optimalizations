#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "experiment_params.hpp"
#include "grid.hpp"
#include <iostream>
#include <random>

namespace infrastructure {

class RandomOnesZerosDataLoader {
  public:
    template <int Dims, typename ElementType>
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) {
        Grid<Dims, ElementType> grid(params.grid_dimensions);

        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_int_distribution<int> dist(0, 1);

        auto grid_data = grid.data();

        for (std::size_t i = 0; i < grid.size(); ++i) {
            grid_data[i] = static_cast<ElementType>(0);
        }

        iterate(grid.as_tile(), [&]() { return static_cast<ElementType>(dist(rng)); });

        return grid;
    }

  private:
    template <int Dims, typename ElementType, typename Func>
    void iterate(infrastructure::GridTile<Dims, ElementType>&& tile, Func&& provide_value) {
        if constexpr (Dims > 1) {
            for (std::size_t i = 1; i < tile.top_dimension_size() - 1; ++i) {
                iterate<Dims - 1, ElementType>(tile[i], provide_value);
            }
        }
        else {
            for (std::size_t i = 1; i < tile.top_dimension_size() - 1; ++i) {
                tile[i] = provide_value();
            }
        }
    }
};

class OneGliderInTheConnerLoader {
  public:
    template <int Dims, typename ElementType>
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) {
        Grid<Dims, ElementType> grid(params.grid_dimensions);

        auto grid_data = grid.data();

        for (std::size_t i = 0; i < grid.size(); ++i) {
            grid_data[i] = static_cast<ElementType>(0);
        }

        if constexpr (Dims == 2) {
            grid[1][2] = 1;
            grid[2][3] = 1;
            grid[3][1] = 1;
            grid[3][2] = 1;
            grid[3][3] = 1;
        }

        return grid;
    }
};

} // namespace infrastructure

#endif // DATA_LOADER_HPP