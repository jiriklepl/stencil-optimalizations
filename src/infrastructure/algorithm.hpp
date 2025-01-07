#ifndef INFRASTRUCTURE_ALGORITHM_HPP
#define INFRASTRUCTURE_ALGORITHM_HPP

#include "experiment_params.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include <cstddef>
#include <iomanip>
#include <memory>

namespace infrastructure {

template <int Dims, typename ElementType>
class Algorithm {
  public:
    using size_type = std::size_t;
    using DataGrid = Grid<Dims, ElementType>;

    /**
     * @brief Sets the input data and formats it if necessary.
     *
     * @param data The input data grid.
     */
    virtual void set_and_format_input_data(const DataGrid& data) = 0;

    /**
     * @brief Initializes data structures and copies data to the device if it is a CUDA algorithm.
     */
    virtual void initialize_data_structures() = 0;

    /**
     * @brief Runs the algorithm for a specified number of iterations.
     *
     * @param iterations The number of iterations to run.
     */
    virtual void run(size_type iterations) = 0;

    /**
     * @brief Copies data back to the host and finalizes it.
     */
    virtual void finalize_data_structures() = 0;

    /**
     * @brief Fetches the result of the algorithm and transforms it back to a grid format.
     *
     * @return DataGrid The result of the algorithm.
     */
    virtual DataGrid fetch_result() = 0;

    /**
     * @brief Sets the experiment parameters.
     *
     * @param params The experiment parameters.
     */
    void set_params(const ExperimentParams& params) {
        this->params = params;
    }

  protected:
    ExperimentParams params;
};

struct TimeReport {
    double set_and_format_input_data = -1;
    double initialize_data_structures = -1;
    double run = -1;
    double finalize_data_structures = -1;
    double fetch_result = -1;

    std::string pretty_print() {
        std::string title_color = "\033[1;34m";
        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[32m";
        std::string reset_color = "\033[0m";

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += labels_color + "  set_and_format_input_data:  " + time_color + std::to_string(set_and_format_input_data) + "s\n";
        result += labels_color + "  initialize_data_structures: " + time_color + std::to_string(initialize_data_structures) + "s\n";
        result += labels_color + "  run:                        " + time_color + std::to_string(run) + "s\n";
        result += labels_color + "  finalize_data_structures:   " + time_color + std::to_string(finalize_data_structures) + "s\n";
        result += labels_color + "  fetch_result:               " + time_color + std::to_string(fetch_result) + "s\n" + reset_color;
        // clang-format on

        return result;
    }

    std::string pretty_print_speedup(const TimeReport& bench) {
        std::string title_color = "\033[1;34m";
        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[34m";
        std::string reset_color = "\033[0m";

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += labels_color + "  set_and_format_input_data:  " + time_color + std::to_string(set_and_format_input_data) + "s ~ " + speedup_str(bench.set_and_format_input_data, set_and_format_input_data) + "\n";
        result += labels_color + "  initialize_data_structures: " + time_color + std::to_string(initialize_data_structures) + "s ~ " + speedup_str(bench.initialize_data_structures, initialize_data_structures) + "\n";
        result += labels_color + "  run:                        " + time_color + std::to_string(run) + "s ~ " + speedup_str(bench.run, run) + "\n";
        result += labels_color + "  finalize_data_structures:   " + time_color + std::to_string(finalize_data_structures) + "s ~ " + speedup_str(bench.finalize_data_structures, finalize_data_structures) + "\n";
        result += labels_color + "  fetch_result:               " + time_color + std::to_string(fetch_result) + "s ~ " + speedup_str(bench.fetch_result, fetch_result) + "\n" + reset_color;
        // clang-format on

        return result;
    }

  private:
    std::string speedup_str(double bench, double time) {
        std::string positive_color = "\033[32m";
        std::string negative_color = "\033[31m";
        std::string reset_color = "\033[0m";

        double speedup = bench / time;

        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << speedup;
        std::string speedup_str = stream.str();

        if (speedup > 1 || speedup_str == "1.00") {
            return positive_color + speedup_str + " x" + reset_color;
        }
        else {
            return negative_color + speedup_str + " x" + reset_color;
        }
    }
};

template <int Dims, typename ElementType>
class TimedAlgorithm : public Algorithm<Dims, ElementType> {
  public:
    using size_type = std::size_t;
    using DataGrid = Grid<Dims, ElementType>;

    TimedAlgorithm(Algorithm<Dims, ElementType>* algorithm) : algorithm(algorithm) {
    }

    void set_and_format_input_data(const DataGrid& data) override {
        Timer t;

        time_report.set_and_format_input_data = t.measure([&]() { algorithm->set_and_format_input_data(data); });
    }

    void initialize_data_structures() override {
        Timer t;

        time_report.initialize_data_structures = t.measure([&]() { algorithm->initialize_data_structures(); });
    }

    void run(size_type iterations) override {
        Timer t;

        time_report.run = t.measure([&]() { algorithm->run(iterations); });
    }

    void finalize_data_structures() override {
        Timer t;

        time_report.finalize_data_structures = t.measure([&]() { algorithm->finalize_data_structures(); });
    }

    DataGrid fetch_result() override {
        Timer t;

        DataGrid result;

        time_report.fetch_result = t.measure([&]() { result = algorithm->fetch_result(); });

        return result;
    }

    TimeReport get_time_report() {
        return time_report;
    }

  private:
    TimeReport time_report;
    Algorithm<Dims, ElementType>* algorithm;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_ALGORITHM_HPP