#ifndef INFRASTRUCTURE_GRID_HPP
#define INFRASTRUCTURE_GRID_HPP

#include <stdexcept>
#include <vector>

namespace infrastructure {

template <int Dims, typename ElementType>
class GridTile {

  public:
    using size_type = std::size_t;
    using LowerDimTile = GridTile<Dims - 1, ElementType>;

    GridTile(ElementType *data, size_type *tile_sizes_per_dimensions)
        : _data(data), tile_sizes_per_dimensions(tile_sizes_per_dimensions) {
    }

    LowerDimTile operator[](int index) {
        auto tile_size = tile_sizes_per_dimensions[0];
        return LowerDimTile(_data + index * tile_size, tile_sizes_per_dimensions + 1);
    }

    ElementType* data() {
        return _data;
    }

  private:
    ElementType *_data;
    size_type *tile_sizes_per_dimensions;
};

template <typename ElementType>
class GridTile<1, ElementType> {
  public:
    using size_type = std::size_t;

    GridTile(ElementType *data, size_type *tile_sizes_per_dimensions) : _data(data) {
    }

    auto &operator[](int index) {
        return _data[index];
    }
    
  private:
    ElementType *_data;
};

template <int DIMS, typename ElementType>
class Grid {

  public:
    using size_type = std::size_t;

    template <typename... Sizes, typename = std::enable_if_t<sizeof...(Sizes) == DIMS>>
    Grid(Sizes... dims) : Grid(std::vector<size_type>{static_cast<size_type>(dims)...}) {
    }

    Grid(const std::vector<size_type> &dimension_sizes) : dimension_sizes(dimension_sizes) {
        if (dimension_sizes.size() != DIMS) {
            throw std::invalid_argument("Dimension sizes must match the number of dimensions");
        }

        size_type total_size = 1;
        tile_sizes_per_dimensions.resize(DIMS);

        for (int i = DIMS - 1; i >= 0; i--) { 
            total_size *= dimension_sizes[i];
            tile_sizes_per_dimensions[i] = total_size;
        }

        elements.resize(total_size);
    }

    auto operator[](size_type index) {
        GridTile<DIMS, ElementType> indexer(elements.data(), tile_sizes_per_dimensions.data() + 1);
        return indexer[index];
    }

    static constexpr int dimensions() {
        return DIMS;
    }

    ElementType* data() {
        return elements.data();
    }

//   private:
    std::vector<ElementType> elements;
    std::vector<size_type> dimension_sizes;
    std::vector<size_type> tile_sizes_per_dimensions;
};

} // namespace infrastructure

#endif