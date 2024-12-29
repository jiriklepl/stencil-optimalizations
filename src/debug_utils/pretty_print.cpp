#include "pretty_print.hpp"

std::string debug_utils::pretty(const infrastructure::Grid<2, char>& grid) {
    std::string result;

    for (int x = 0; x < grid.size_in<0>(); ++x) {
        for (int y = 0; y < grid.size_in<1>(); ++y) {
            // result += (grid[x][y] == 1 ? "O" : ".");
            std::string cell = "[";
            cell += std::to_string(static_cast<int>(grid[x][y]));
            cell += ']';

            if (grid[x][y] != 0) {
                result += "\033[31m" + cell + "\033[0m"; // Red color
            }
            else {
                result += cell;
            }
        }
        result += "\n";
    }

    return result;
}

std::string debug_utils::pretty(const infrastructure::Grid<3, char>& grid) {
    std::string result;
    auto dimX = grid.size_in<0>();
    auto dimY = grid.size_in<1>();
    auto dimZ = grid.size_in<2>();

    for (int z = 0; z < dimZ; z++) {
        result += "Layer Z=" + std::to_string(z) + "\n";
        for (int x = 0; x < dimX; ++x) {
            for (int y = 0; y < dimY; ++y) {
                result += (grid[x][y][z] == 1 ? "O" : ".");
            }
            result += "\n";
        }
        result += "\n";
    }

    return result;
}
