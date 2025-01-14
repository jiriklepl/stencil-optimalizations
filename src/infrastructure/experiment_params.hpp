#ifndef INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
#define INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include "colors.hpp"

namespace infrastructure {

enum class StreamingDirection {
    in_X = 0,
    in_Y = 1,
    NAIVE = 2,
};

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
    bool colorful = true;

    std::size_t random_seed = 42;

    std::size_t thread_block_size;

    std::size_t warp_dims_x;
    std::size_t warp_dims_y;
    std::size_t warp_tile_dims_x;
    std::size_t warp_tile_dims_y;

    StreamingDirection streaming_direction;

    std::string pretty_print() {
      std::stringstream ss;

      std::string title_color = c::title_color();
      std::string label_color = c::label_color();
      std::string value_color = c::value_color();
      std::string reset_color = c::reset_color();

      ss << title_color << "Experiment Parameter:" << std::endl;

      ss << label_color << "  algorithm_name: " << value_color << algorithm_name << std::endl;
      ss << label_color << "  grid_dimensions: " << value_color << grid_dimensions[0] << "x" << grid_dimensions[1] << std::endl;
      ss << label_color << "  iterations: " << value_color << iterations << std::endl << std::endl;
      ss << label_color << "  data_loader_name: " << value_color << data_loader_name << std::endl;
      ss << label_color << "  pattern_expression: " << value_color << pattern_expression << std::endl << std::endl;
      ss << label_color << "  measure_speedup: " << value_color << measure_speedup << std::endl;
      ss << label_color << "  speedup_bench_algorithm_name: " << value_color << speedup_bench_algorithm_name << std::endl << std::endl;
      ss << label_color << "  validate: " << value_color << validate << std::endl;
      ss << label_color << "  print_validation_diff: " << value_color << print_validation_diff << std::endl;
      ss << label_color << "  validation_algorithm_name: " << value_color << validation_algorithm_name << std::endl << std::endl;
      ss << label_color << "  animate_output: " << value_color << animate_output << std::endl;
      ss << label_color << "  colorful: " << value_color << colorful << std::endl << std::endl;
      ss << label_color << "  random_seed: " << value_color << random_seed << std::endl << std::endl;
      ss << label_color << "  thread_block_size: " << value_color << thread_block_size << std::endl << std::endl;
      ss << label_color << "  warp_dims_x: " << value_color << warp_dims_x << std::endl;
      ss << label_color << "  warp_dims_y: " << value_color << warp_dims_y << std::endl << std::endl;
      ss << label_color << "  warp_tile_dims_x: " << value_color << warp_tile_dims_x << std::endl;
      ss << label_color << "  warp_tile_dims_y: " << value_color << warp_tile_dims_y << std::endl << std::endl;
      ss << label_color << "  streaming_direction: " << value_color;
      switch (streaming_direction) {
        case StreamingDirection::in_X: ss << "in-x"; break;
        case StreamingDirection::in_Y: ss << "in-y"; break;
        case StreamingDirection::NAIVE: ss << "naive"; break;
      }
      ss << reset_color << std::endl;

      return ss.str();
    }
};

namespace opts {
// clang-format off
const std::string ALGORITHM_NAME               = "algorithm";
const std::string GRID_DIMENSIONS_X            = "grid-dimensions-x";
const std::string GRID_DIMENSIONS_Y            = "grid-dimensions-y";
const std::string ITERATIONS                   = "iterations";
const std::string DATA_LOADER_NAME             = "data-loader";
const std::string PATTERN_EXPRESSION           = "pattern-expression";
const std::string MEASURE_SPEEDUP              = "measure-speedup";
const std::string SPEEDUP_BENCH_ALGORITHM_NAME = "speedup-bench-algorithm";
const std::string VALIDATE                     = "validate";
const std::string PRINT_VALIDATION_DIFF        = "print-validation-diff";
const std::string VALIDATION_ALGORITHM_NAME    = "validation-algorithm";
const std::string ANIMATE_OUTPUT               = "animate-output";
const std::string COLORFUL                     = "colorful";
const std::string RANDOM_SEED                  = "random-seed";
const std::string THREAD_BLOCK_SIZE            = "thread-block-size";
const std::string WARP_DIMS_X                  = "warp-dims-x";
const std::string WARP_DIMS_Y                  = "warp-dims-y";
const std::string WARP_TILE_DIMS_X             = "warp-tile-dims-x";
const std::string WARP_TILE_DIMS_Y             = "warp-tile-dims-y";
const std::string STREAMING_DIRECTION          = "streaming-direction";
// clang-format on
}

class ParamsParser {

  public:
    static ExperimentParams parse(int argc, char** argv) {
      cxxopts::Options optConfig("game of life", "");

      optConfig.add_options()
        ("help", "Show help")

        (opts::ALGORITHM_NAME, "Algorithm name",
          cxxopts::value<std::string>())

        (opts::GRID_DIMENSIONS_X, "Grid X dimension",
          cxxopts::value<std::size_t>())
        
        (opts::GRID_DIMENSIONS_Y, "Grid Y dimension",
          cxxopts::value<std::size_t>())
        
        (opts::ITERATIONS, "Number of iterations",
          cxxopts::value<std::size_t>())
        
        (opts::DATA_LOADER_NAME, "Data loader",
          cxxopts::value<std::string>()->default_value("random-ones-zeros"))
        
        (opts::PATTERN_EXPRESSION, "Pattern expression",
          cxxopts::value<std::string>()->default_value(""))
        
        (opts::MEASURE_SPEEDUP, "Measure speedup",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::SPEEDUP_BENCH_ALGORITHM_NAME, "Speedup bench algorithm",
          cxxopts::value<std::string>()->default_value("gol-cpu-naive"))
        
        (opts::VALIDATE, "Validate",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::PRINT_VALIDATION_DIFF, "Print validation diff",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::VALIDATION_ALGORITHM_NAME, "Validation algorithm",
          cxxopts::value<std::string>()->default_value("gol-cpu-naive"))
        
        (opts::ANIMATE_OUTPUT, "Animate output",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::COLORFUL, "Use colorful output",
          cxxopts::value<bool>()->default_value("true"))
        
        (opts::RANDOM_SEED, "Random seed",
          cxxopts::value<std::size_t>()->default_value("42"))
        
        (opts::THREAD_BLOCK_SIZE, "Thread block size",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_DIMS_X, "Warp dimension X",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_DIMS_Y, "Warp dimension Y",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_TILE_DIMS_X, "Warp tile dimension X",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_TILE_DIMS_Y, "Warp tile dimension Y",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::STREAMING_DIRECTION, "Streaming direction (in-x|in-y|naive)",
          cxxopts::value<std::string>()->default_value("naive"))
      ;

      auto result = optConfig.parse(argc, argv);

      if (result.count("help")) {
        std::cout << optConfig.help() << std::endl;
        exit(0);
      }

      ExperimentParams params;
      params.algorithm_name = result[opts::ALGORITHM_NAME].as<std::string>();
      std::size_t gx = result[opts::GRID_DIMENSIONS_X].as<std::size_t>();
      std::size_t gy = result[opts::GRID_DIMENSIONS_Y].as<std::size_t>();

      params.grid_dimensions = {gx, gy};
      params.iterations = result[opts::ITERATIONS].as<std::size_t>();

      params.data_loader_name = result[opts::DATA_LOADER_NAME].as<std::string>();
      params.pattern_expression = result[opts::PATTERN_EXPRESSION].as<std::string>();

      params.measure_speedup = result[opts::MEASURE_SPEEDUP].as<bool>();
      params.speedup_bench_algorithm_name =
        result[opts::SPEEDUP_BENCH_ALGORITHM_NAME].as<std::string>();

      params.validate = result[opts::VALIDATE].as<bool>();
      params.print_validation_diff = result[opts::PRINT_VALIDATION_DIFF].as<bool>();
      params.validation_algorithm_name =
        result[opts::VALIDATION_ALGORITHM_NAME].as<std::string>();

      params.animate_output = result[opts::ANIMATE_OUTPUT].as<bool>();
      params.colorful = result[opts::COLORFUL].as<bool>();
      params.random_seed = result[opts::RANDOM_SEED].as<std::size_t>();

      params.thread_block_size = result[opts::THREAD_BLOCK_SIZE].as<std::size_t>();

      params.warp_dims_x = result[opts::WARP_DIMS_X].as<std::size_t>();
      params.warp_dims_y = result[opts::WARP_DIMS_Y].as<std::size_t>();

      params.warp_tile_dims_x = result[opts::WARP_TILE_DIMS_X].as<std::size_t>();
      params.warp_tile_dims_y = result[opts::WARP_TILE_DIMS_Y].as<std::size_t>();

      auto dir_str = result[opts::STREAMING_DIRECTION].as<std::string>();

      if (dir_str == "in-x")       { params.streaming_direction = StreamingDirection::in_X; }
      else if (dir_str == "in-y")  { params.streaming_direction = StreamingDirection::in_Y; }
      else if (dir_str == "naive") { params.streaming_direction = StreamingDirection::NAIVE; }
      else                         { throw std::runtime_error("Invalid streaming direction"); }

      params.colorful = result[opts::COLORFUL].as<bool>();

      return params;
    }

};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
