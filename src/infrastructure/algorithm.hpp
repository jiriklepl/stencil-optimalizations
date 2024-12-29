#ifndef INFRASTRUCTURE_ALGORITHM_HPP
#define INFRASTRUCTURE_ALGORITHM_HPP

#include "grid.hpp"
#include <cstddef>

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
    virtual DataGrid fetch_result();
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_ALGORITHM_HPP