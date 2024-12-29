#ifndef INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP
#define INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP

#include "algorithm.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace infrastructure {

template <int Dims, typename ElementType>
class AlgorithmRepository {
  public:
    using AlgType = Algorithm<Dims, ElementType>;

    void register_algorithm(const std::string& algorithm_name, std::unique_ptr<AlgType> algorithm) {
        _algorithms[algorithm_name] = std::move(algorithm);
    }

    AlgType* fetch_algorithm(const std::string& algorithm_name) {
        auto it = _algorithms.find(algorithm_name);
        if (it != _algorithms.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    bool has_algorithm(const std::string& algorithm_name) {
        return _algorithms.find(algorithm_name) != _algorithms.end();
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<AlgType>> _algorithms;
};
} // namespace infrastructure

#endif // INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP