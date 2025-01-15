#ifndef TIMER_HPP
#define TIMER_HPP

#include "../algorithms/_shared/cuda-helpers/cuch.hpp"
#include <chrono>
#include <cuda_runtime.h>

namespace infrastructure {

class Timer {

  public:
    template <typename Func>
    static double measure(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();

        func();
        CUCH(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double>(end - start).count();
    }
};

class StopWatch {
  public:
    StopWatch(std::size_t count_down_seconds) : count_down_seconds(count_down_seconds) {
        restart();
    }

    void restart() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    bool time_is_up() {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration<double>(now - start_time).count();

        return elapsed_seconds >= count_down_seconds;
    }

  private:
    std::size_t count_down_seconds;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};


} // namespace infrastructure

#endif // TIMER_HPP