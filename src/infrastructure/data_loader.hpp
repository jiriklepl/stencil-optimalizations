#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "experiment_params.hpp"
#include "grid.hpp"
#include <iostream>
#include <memory>
#include <random>

namespace infrastructure {

template <int Dims, typename ElementType>
class Loader {
  public:
    virtual Grid<Dims, ElementType> load_data(const ExperimentParams& params) = 0;

    virtual std::unique_ptr<Grid<Dims, ElementType>> load_validation_data(const ExperimentParams& params) {
        return nullptr;
    }
};

template <int Dims, typename ElementType>
class LoaderCtorBase {
  public:
    virtual std::unique_ptr<Loader<Dims, ElementType>> create() = 0;
};

template <template <int Dims, typename ElementType> class LoaderType, int Dims, typename ElementType>
class LoaderCtor : public LoaderCtorBase<Dims, ElementType> {
  public:
    std::unique_ptr<Loader<Dims, ElementType>> create() override {
        return std::make_unique<LoaderType<Dims, ElementType>>();
    }
};

template <int Dims, typename ElementType>
class RandomOnesZerosDataLoader : public Loader<Dims, ElementType> {
  public:
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) override {
        Grid<Dims, ElementType> grid(params.grid_dimensions);

        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_int_distribution<int> dist(0, 1);

        auto grid_data = grid.data();

        for (std::size_t i = 0; i < grid.size(); ++i) {
            grid_data[i] = static_cast<ElementType>(dist(rng));
            ;
        }

        return grid;
    }
};

template <int Dims, typename ElementType>
class OneGliderInTheConnerLoader : public Loader<Dims, ElementType> {
  public:
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) {
        Grid<Dims, ElementType> grid(params.grid_dimensions);

        auto grid_data = grid.data();

        for (std::size_t i = 0; i < grid.size(); ++i) {
            grid_data[i] = static_cast<ElementType>(0);
        }

        // if constexpr (Dims == 2) {
        //     grid[1][2] = 1;
        //     grid[2][3] = 1;
        //     grid[3][1] = 1;
        //     grid[3][2] = 1;
        //     grid[3][3] = 1;
        // }

        if constexpr (Dims == 2) {
            grid[0][1] = 1;
            grid[1][2] = 1;
            grid[2][0] = 1;
            grid[2][1] = 1;
            grid[2][2] = 1;
        }

        return grid;
    }
};

} // namespace infrastructure

#endif // DATA_LOADER_HPP