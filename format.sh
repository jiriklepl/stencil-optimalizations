find src/ \( -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' \) ! -name 'bitwise_ops_macros.hpp' | xargs clang-format -i
