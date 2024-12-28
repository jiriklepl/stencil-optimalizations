#define DEBUG

#include <iostream>
#include "infrastructure/grid.hpp"
#include "debug_utils/pretty_print.hpp"

using namespace debug_utils;

int main() {
    std::cout << "Hello, world!" << std::endl;

    infrastructure::Grid<3, int> grid(2, 3, 4);

    auto counter = 1;
    for (auto& elem : grid.elements) {
        elem = counter++;
    }

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 4; k++) {
                auto x = &grid[i][j][k];
                
                std::cout << x << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << pretty(grid.dimension_sizes) << std::endl;
    std::cout << pretty(grid.tile_sizes_per_dimensions) << std::endl;
    std::cout << pretty(grid.elements) << std::endl;

    std::cout << "ended" << std::endl;

    return 0;
}