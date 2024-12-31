#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/grid.hpp"
#define DEBUG

#include <iostream>

#include "debug_utils/pretty_print.hpp"
using namespace debug_utils;

int main() {
    std::cout << "Hello" << std::endl;

    infrastructure::ExperimentParams params = {
        .algorithm_name = "gol-cpu-naive",
        .grid_dimensions = {10, 20},
        .iterations = 50,
    };

    infrastructure::ExperimentManager manager;

    manager.run(params);

    return 0;
}