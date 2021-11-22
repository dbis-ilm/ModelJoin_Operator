file(GLOB_RECURSE ALL_TEST_C *_test.c)

foreach(ONE_TEST_C ${ALL_TEST_C})
    get_filename_component(ONE_TEST_EXEC ${ONE_TEST_C} NAME_WE)
    add_executable (${ONE_TEST_EXEC} ${ONE_TEST_C})
    target_link_libraries(${ONE_TEST_EXEC}  PUBLIC Src Ops Utils ${CMAKE_DL_LIBS} m)
    add_test (${ONE_TEST_EXEC} ${EXECUTABLE_OUTPUT_PATH}/${ONE_TEST_EXEC})
endforeach()
