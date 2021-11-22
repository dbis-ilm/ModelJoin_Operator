add_subdirectory(Modeljoin)

add_library(Ops scan.c compare.c)
target_link_libraries(Ops Src ML_layers)