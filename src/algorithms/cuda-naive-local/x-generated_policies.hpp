#ifndef GEN_POLICIES_HPP
#define GEN_POLICIES_HPP

#include <cstddef>
#include <cstdint>

namespace algorithms {

template <typename const_t>
struct thb1024__warp32x1__warp_tile32x1 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 1;
};

template <typename const_t>
struct thb1024__warp32x1__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb1024__warp32x1__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb1024__warp32x1__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb1024__warp32x1__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb1024__warp16x2__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb1024__warp16x2__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb1024__warp16x2__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb1024__warp16x2__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 1024;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb512__warp32x1__warp_tile32x1 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 1;
};

template <typename const_t>
struct thb512__warp32x1__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb512__warp32x1__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb512__warp32x1__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb512__warp32x1__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb512__warp16x2__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb512__warp16x2__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb512__warp16x2__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb512__warp16x2__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 512;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x1 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 1;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb256__warp32x1__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb256__warp16x2__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb256__warp16x2__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb256__warp16x2__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb256__warp16x2__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 256;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb128__warp32x1__warp_tile32x1 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 1;
};

template <typename const_t>
struct thb128__warp32x1__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb128__warp32x1__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb128__warp32x1__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb128__warp32x1__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 32;
    constexpr static const_t WARP_DIM_Y = 1;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};

template <typename const_t>
struct thb128__warp16x2__warp_tile32x8 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 8;
};

template <typename const_t>
struct thb128__warp16x2__warp_tile32x16 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 16;
};

template <typename const_t>
struct thb128__warp16x2__warp_tile32x32 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 32;
};

template <typename const_t>
struct thb128__warp16x2__warp_tile32x64 {
    constexpr static const_t THREAD_BLOCK_SIZE = 128;

    constexpr static const_t WARP_DIM_X = 16;
    constexpr static const_t WARP_DIM_Y = 2;

    constexpr static const_t WARP_TILE_DIM_X = 32;
    constexpr static const_t WARP_TILE_DIM_Y = 64;
};


}

#endif // GEN_POLICIES_HPP