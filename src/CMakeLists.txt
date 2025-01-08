enable_language(CUDA)

# Gather CUDA source files
file(GLOB_RECURSE CudaFiles
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Gather regular C++ source files
file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set_source_files_properties(algorithms/an5d/generated-edited/gol_32_an5d_host.cu PROPERTIES COMPILE_FLAGS "-w")
set_source_files_properties(algorithms/an5d/generated-edited/gol_32_an5d_kernel.cu PROPERTIES COMPILE_FLAGS "-w")
set_source_files_properties(algorithms/an5d/generated-edited/gol_64_an5d_host.cu PROPERTIES COMPILE_FLAGS "-w")
set_source_files_properties(algorithms/an5d/generated-edited/gol_64_an5d_kernel.cu PROPERTIES COMPILE_FLAGS "-w")

add_executable(stencils
    ${SOURCE_FILES}
    ${CudaFiles}
)
