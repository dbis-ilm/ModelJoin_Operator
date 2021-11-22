add_library(Src operator.c buffer.c engine.c profile.c type.c)

if(USE_CUDA MATCHES ON) 
    target_link_libraries(Src cuda_utils)
endif()