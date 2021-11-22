add_library(Utils stack.c)

if(USE_CUDA MATCHES ON) 
    add_library(cuda_utils cuda_utils.cu)
    target_link_libraries(cuda_utils cublas)
endif()