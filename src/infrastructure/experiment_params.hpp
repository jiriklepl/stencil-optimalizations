#ifndef INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
#define INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP

#include <cstddef>
#include <string>
#include <vector>
namespace infrastructure {

class ExperimentParams {
  public:
    std::string algorithm_name;
    std::vector<std::size_t> grid_dimensions;
    std::size_t random_seed = 42;
    std::size_t iterations;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
