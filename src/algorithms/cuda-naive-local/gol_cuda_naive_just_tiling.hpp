#ifndef GOL_CUDA_NAIVE_JUST_TILING_HPP
#define GOL_CUDA_NAIVE_JUST_TILING_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"
#include "./models.hpp"
#include "./tiling-policies.cuh"
#include "./x-generated_policies.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

namespace algorithms::cuda_naive_local {

using StreamingDir = infrastructure::StreamingDirection;

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
class GoLCudaNaiveJustTiling : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
    GoLCudaNaiveJustTiling() = default;

    using size_type = std::size_t;
    using word_type = typename BitsConst<Bits>::word_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = GeneralBitGrid<word_type, bit_grid_mode>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    template <template <typename> class base_policy>
    using policy = extended_policy<size_type, base_policy>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        thread_block_size = this->params.thread_block_size;

        assert(warp_size() == 32);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();
        
        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(word_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(word_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(word_type), cudaMemcpyHostToDevice));

    }

    void run(size_type iterations) override {
        switch (this->params.streaming_direction) {
            case StreamingDir::in_X:
                run_kernel_in_direction<StreamingDir::in_X>(iterations);
                break;
            case StreamingDir::in_Y:
                run_kernel_in_direction<StreamingDir::in_Y>(iterations);
                break;
            case StreamingDir::NAIVE:
                run_kernel_in_direction<StreamingDir::NAIVE>(iterations);
                break;
            default:
                throw std::runtime_error("Invalid streaming direction");
        }
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(word_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridWithTiling<word_type> cuda_data;

    std::size_t thread_block_size;
    std::size_t _performed_iterations;

    template <StreamingDir Direction, typename tiling_policy>
    void run_kernel(size_type iterations);

    std::size_t warp_tile_size() const {
        return this->params.warp_tile_dims_x * this->params.warp_tile_dims_y;
    }

    std::size_t warp_size() const {
        return this->params.warp_dims_x * this->params.warp_dims_y;
    }

    std::size_t get_warp_tiles_count() const {
        return bit_grid->size() / warp_tile_size();
    }

    std::size_t get_thread_block_count() const {
        const auto warps_per_block = thread_block_size / warp_size();
        const auto computed_elems_in_block = warps_per_block * warp_tile_size(); 

        return bit_grid->size() / computed_elems_in_block;
    }

    std::size_t state_store_word_count() const {
        return get_thread_block_count();
    }

    std::size_t tiles_per_block() const {
        return thread_block_size / warp_size();
    }

    template <StreamingDir Direction>
    void run_kernel_in_direction(size_type iterations) {
        if (policy<thb1024__warp32x1__warp_tile32x1>::is_for(this->params)) {
            run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x1>>(iterations);
        }
        // else if (policy<thb1024__warp32x1__warp_tile32x8>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x8>>(iterations);
        // }
        // else if (policy<thb1024__warp32x1__warp_tile32x16>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x16>>(iterations);
        // }
        // else if (policy<thb1024__warp32x1__warp_tile32x32>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x32>>(iterations);
        // }
        // else if (policy<thb1024__warp32x1__warp_tile32x64>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x64>>(iterations);
        // }
        // else if (policy<thb1024__warp16x2__warp_tile32x8>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x8>>(iterations);
        // }
        // else if (policy<thb1024__warp16x2__warp_tile32x16>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x16>>(iterations);
        // }
        // else if (policy<thb1024__warp16x2__warp_tile32x32>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x32>>(iterations);
        // }
        // else if (policy<thb1024__warp16x2__warp_tile32x64>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x64>>(iterations);
        // }
        // else if (policy<thb512__warp32x1__warp_tile32x1>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x1>>(iterations);
        // }
        // else if (policy<thb512__warp32x1__warp_tile32x8>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x8>>(iterations);
        // }
        // else if (policy<thb512__warp32x1__warp_tile32x16>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x16>>(iterations);
        // }
        else if (policy<thb512__warp32x1__warp_tile32x32>::is_for(this->params)) {
            run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x32>>(iterations);
        }
        else if (policy<thb512__warp32x1__warp_tile32x64>::is_for(this->params)) {
            run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x64>>(iterations);
        }
        // else if (policy<thb512__warp16x2__warp_tile32x8>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x8>>(iterations);
        // }
        // else if (policy<thb512__warp16x2__warp_tile32x16>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x16>>(iterations);
        // }
        // else if (policy<thb512__warp16x2__warp_tile32x32>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x32>>(iterations);
        // }
        // else if (policy<thb512__warp16x2__warp_tile32x64>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x64>>(iterations);
        // }
        else if (policy<thb256__warp32x1__warp_tile32x1>::is_for(this->params)) {
            run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x1>>(iterations);
        }
        else if (policy<thb256__warp32x1__warp_tile32x8>::is_for(this->params)) {
            run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x8>>(iterations);
        }
        else if (policy<thb256__warp32x1__warp_tile32x16>::is_for(this->params)) {
            run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x16>>(iterations);
        }
        else if (policy<thb256__warp32x1__warp_tile32x32>::is_for(this->params)) {
            run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x32>>(iterations);
        }
        // else if (policy<thb256__warp32x1__warp_tile32x64>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x64>>(iterations);
        // }
        // else if (policy<thb256__warp16x2__warp_tile32x8>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x8>>(iterations);
        // }
        // else if (policy<thb256__warp16x2__warp_tile32x16>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x16>>(iterations);
        // }
        // else if (policy<thb256__warp16x2__warp_tile32x32>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x32>>(iterations);
        // }
        // else if (policy<thb256__warp16x2__warp_tile32x64>::is_for(this->params)) {
        //     run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x64>>(iterations);
        // }
        else {
            throw std::runtime_error("Invalid policy");
        }
    }
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP