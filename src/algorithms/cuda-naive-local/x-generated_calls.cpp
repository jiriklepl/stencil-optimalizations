// #include "./gol_cuda_naive_just_tiling.hpp"
// #include "./gol_cuda_naive_local.hpp"
// #include "./x-generated_policies.hpp"

// namespace algorithms::cuda_naive_local {

// template <template <typename> class base_policy>
// using policy = extended_policy<std::size_t, base_policy>;


// template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
// template <StreamingDir Direction>
// void GoLCudaNaiveJustTiling<grid_cell_t, Bits, bit_grid_mode>::run_kernel_in_direction(
//     size_type iterations){

//     if (policy<thb1024__warp32x1__warp_tile32x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x1>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x1>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb1024__warp32x1__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp32x1__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb1024__warp16x2__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp16x2__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb1024__warp8x4__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb1024__warp8x4__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x1>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x1>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb512__warp32x1__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp32x1__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb512__warp16x2__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp16x2__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb512__warp8x4__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb512__warp8x4__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x1>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x1>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb256__warp32x1__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp32x1__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb256__warp16x2__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp16x2__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb256__warp8x4__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb256__warp8x4__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x1>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x1>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb128__warp32x1__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp32x1__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb128__warp16x2__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp16x2__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb128__warp8x4__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb128__warp8x4__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x1>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x1>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x1>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb64__warp32x1__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp32x1__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile32x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile32x2>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile64x2>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile64x2>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb64__warp16x2__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp16x2__warp_tile64x32>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile32x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile32x4>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile32x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile32x8>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile32x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile32x16>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile32x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile32x32>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile64x4>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile64x4>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile64x8>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile64x8>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile64x16>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile64x16>>(iterations);
//     }
//     else if (policy<thb64__warp8x4__warp_tile64x32>::is_for(this->params)) {
//         run_kernel<Direction, policy<thb64__warp8x4__warp_tile64x32>>(iterations);
//     }
//     else {
//         throw std::runtime_error("Invalid policy");
//     }
// }

// }