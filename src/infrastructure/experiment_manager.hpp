#ifndef INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP
#define INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP

#include "../algorithms/cpu-naive/gol_cpu_naive.hpp"
#include "algorithm_repository.hpp"
#include "data_loader.hpp"
#include "experiment_params.hpp"
#include <iostream>
#include <memory>

#include "../debug_utils/pretty_print.hpp"
using namespace debug_utils;

namespace infrastructure {

class ExperimentManager {
  public:
    ExperimentManager() {
        auto cpu_naive = std::make_unique<algorithms::GoLCpuNaive>();
        cpu_naive->print_game_of_live_in_progress();

        _algs_repo_2d_char.register_algorithm("gol-cpu-naive", std::move(cpu_naive));
        _algs_repo_2d_char.register_algorithm("gol-cpu-bitwise", nullptr);
    }

    void run(const ExperimentParams& params) {
        for_each_repo([&](auto& repo) {
            if (repo.has_algorithm(params.algorithm_name)) {
                run_experiment(repo, params);
            }
        });
    }

  private:
    template <int Dims, typename ElementType>
    void run_experiment(AlgorithmRepository<Dims, ElementType>& repo, const ExperimentParams& params) {
        auto data = load_data<Dims, ElementType>(params);
        auto alg = repo.fetch_algorithm(params.algorithm_name);

        alg->set_and_format_input_data(data);
        alg->initialize_data_structures();
        alg->run(params.iterations);
        alg->finalize_data_structures();

        auto result = alg->fetch_result();
        // std::cout << pretty(result) << std::endl;
    }

    template <typename F>
    void for_each_repo(F&& f) {
        f(_algs_repo_2d_char);
        f(_algs_repo_3d_char);
    }

    template <int Dims, typename ElementType>
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) {
        // TODO: file loading etc

        // RandomOnesZerosDataLoader loader;
        // return loader.load_data<Dims, ElementType>(params);

        OneGliderInTheConnerLoader loader;
        return loader.load_data<Dims, ElementType>(params);
    }

    AlgorithmRepository<2, char> _algs_repo_2d_char;
    AlgorithmRepository<3, char> _algs_repo_3d_char;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP