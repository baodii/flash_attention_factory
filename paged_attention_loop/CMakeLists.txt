set(LIB_NAME paged_attention_loop)

add_library(${LIB_NAME} SHARED paged_attention_loop.cpp)

set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")

target_compile_definitions(${LIB_NAME} PRIVATE TORCH_EXTENSION_NAME=${LIB_NAME})
message(STATUS "AOT_FLAGS: ${AOT_FLAGS}")
# target_compile_options(${LIB_NAME} PRIVATE ${AOT_FLAGS})
target_link_libraries(${LIB_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_options(${LIB_NAME} PRIVATE -Wl,-rpath,${PROJECT_BINARY_DIR} ${AOT_FLAGS})

target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Copy final .so to lib/ as Python extension
add_custom_command(TARGET ${LIB_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${LIB_NAME}>
            ${LIB_OUTPUT_DIR}/
)
