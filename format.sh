find src/ \( -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' \) \
    ! -name 'bitwise_ops_macros.hpp' \
    ! -name 'cpu_gol_32_64.cpp' \
    ! -name 'gol_32_an5d_host.cu' \
    ! -name 'gol_32_an5d_kernel.cu' \
    ! -name 'gol_32_an5d_kernel.hu' \
    ! -name 'gol_64_an5d_host.cu' \
    ! -name 'gol_64_an5d_kernel.cu' \
    ! -name 'gol_64_an5d_kernel.hu' \
    ! -name 'patterns.hpp' \
      -print0 \
    | xargs -0 clang-format -i
