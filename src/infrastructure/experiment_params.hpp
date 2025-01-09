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
    std::size_t iterations;

    std::string data_loader_name;
    std::string pattern_expression;

    bool measure_speedup = false;
    std::string speedup_bench_algorithm_name;

    bool validate = false;
    bool print_validation_diff = false;

    std::string validation_algorithm_name = "gol-cpu-naive";

    bool animate_output = false;

    std::size_t random_seed = 42;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
