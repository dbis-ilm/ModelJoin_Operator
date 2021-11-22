cmake_minimum_required(VERSION 3.10)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()


#options
option(USE_CUDA    "Use CUDA"         OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_definitions(-DPROFILE)

if(USE_CUDA MATCHES ON) 
  add_definitions(-DUSECUDA)
  enable_language(CUDA)
  if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
  endif()
endif()

# set the project name
project(SimpleCStore)

# print_all_variables()


# add the libraries
add_subdirectory(src)
add_subdirectory(src/ops)
add_subdirectory(src/utils)

# add the executable
add_executable(example main.c)

target_link_libraries(example PUBLIC Src Ops Utils ${CMAKE_DL_LIBS} m)

enable_testing()
add_subdirectory(test)
