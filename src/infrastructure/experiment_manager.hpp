#ifndef INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP
#define INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP

#include "../algorithms/cpu-bitwise-cols/gol_cpu_bitwise_cols.hpp"
#include "../algorithms/cpu-naive/gol_cpu_naive.hpp"
#include "./data_loader.hpp"
#include "algorithm.hpp"
#include "algorithm_repository.hpp"
#include "data_loader.hpp"
#include "experiment_params.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "../debug_utils/diff_grid.hpp"
#include "../debug_utils/pretty_print.hpp"
#include "grid.hpp"
using namespace debug_utils;

namespace infrastructure {

class ExperimentManager {
    template <int Dims, typename ElementType>
    using grid_ptr = std::unique_ptr<Grid<Dims, ElementType>>;

    enum class AlgMode {
        Timed = 0,
        NotTimed = 1,
    };

  public:
    ExperimentManager() {
        auto cpu_naive = std::make_unique<algorithms::GoLCpuNaive>();
        auto cpu_bitwise_16 = std::make_unique<algorithms::GoLCpuBitwiseCols<16>>();
        auto cpu_bitwise_32 = std::make_unique<algorithms::GoLCpuBitwiseCols<32>>();
        auto cpu_bitwise_64 = std::make_unique<algorithms::GoLCpuBitwiseCols<64>>();

        _algs_repos.get_repository<2, char>()->register_algorithm("gol-cpu-naive", std::move(cpu_naive));
        _algs_repos.get_repository<2, char>()->register_algorithm("gol-cpu-bitwise-cols-16", std::move(cpu_bitwise_16));
        _algs_repos.get_repository<2, char>()->register_algorithm("gol-cpu-bitwise-cols-32", std::move(cpu_bitwise_32));
        _algs_repos.get_repository<2, char>()->register_algorithm("gol-cpu-bitwise-cols-64", std::move(cpu_bitwise_64));
    }

    void run(const ExperimentParams& params) {
        _algs_repos.for_each([&](auto& repo) {
            if (repo.has_algorithm(params.algorithm_name)) {
                run_experiment(repo, params);
            }
        });
    }

  private:
    template <int Dims, typename ElementType>
    void run_experiment(AlgorithmRepository<Dims, ElementType>& repo, const ExperimentParams& params) {

        auto loader = load_data<Dims, ElementType>(params);
        auto data = loader->load_data(params);
        auto alg = repo.fetch_algorithm(params.algorithm_name);

        auto [result, time_report] = perform_alg<AlgMode::Timed>(*alg, data, params);

        std::cout << time_report.pretty_print() << std::endl;

        if (params.validate) {
            validate(data, *result.get(), *loader.get(), params);
        }
    }

    template <int Dims, typename ElementType>
    void validate(const Grid<Dims, ElementType>& original, const Grid<Dims, ElementType>& result,
                  Loader<Dims, ElementType>& loader, const ExperimentParams& params) {

        std::cout << "\033[35mValidating result...\033[0m" << std::endl;

        auto validation_data = loader.load_validation_data(params);

        if (validation_data == nullptr) {
            auto repo = _algs_repos.template get_repository<Dims, ElementType>();
            auto alg = repo->fetch_algorithm(params.validation_algorithm_name);

            validation_data = perform_alg<AlgMode::NotTimed>(*alg, original, params);
        }

        if (validation_data->equals(result)) {
            std::cout << "\033[32mValidation successful\033[0m" << std::endl;
        }
        else {
            std::cout << "\033[31mValidation failed\033[0m" << std::endl;

            if (params.print_validation_diff) {
                auto diff = debug_utils::diff(*validation_data.get(), result);
                std::cout << "Diff: \n" << diff << std::endl;
            }
        }
    }

    template <AlgMode mode, int Dims, typename ElementType>
    auto perform_alg(Algorithm<Dims, ElementType>& alg, const Grid<Dims, ElementType>& init_data,
                     const ExperimentParams& params) {
        TimedAlgorithm<Dims, ElementType> timed_alg(&alg);

        timed_alg.set_params(params);

        timed_alg.set_and_format_input_data(init_data);
        timed_alg.initialize_data_structures();
        timed_alg.run(params.iterations);
        timed_alg.finalize_data_structures();

        auto result = timed_alg.fetch_result();
        auto result_ptr = std::make_unique<Grid<Dims, ElementType>>(result);

        auto time_report = timed_alg.get_time_report();

        if constexpr (mode == AlgMode::Timed) {
            return std::make_tuple(std::move(result_ptr), time_report);
        }
        else {
            return std::move(result_ptr);
        }
    }

    template <int Dims, typename ElementType>
    std::unique_ptr<Loader<Dims, ElementType>> load_data(const ExperimentParams& params) {
        LoaderCtor<RandomOnesZerosDataLoader, Dims, ElementType> random_loader;
        LoaderCtor<OneGliderInTheConnerLoader, Dims, ElementType> one_glider_loader;

        auto loaderCtor = std::map<std::string, LoaderCtorBase<Dims, ElementType>*>{
            {"random-ones-zeros", &random_loader},
            {"one-glider-in-the-conner", &one_glider_loader},
        }[params.data_loader_name];

        return loaderCtor->create();
    }

    AlgorithmReposCollection<AlgRepoParams<2, char>, AlgRepoParams<3, char>> _algs_repos;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP