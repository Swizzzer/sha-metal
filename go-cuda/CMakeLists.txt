cmake_minimum_required(VERSION 3.18)

project(sha1-crack LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_TOOLKIT_ROOT}")

find_program(GO_EXECUTABLE go REQUIRED)
message(STATUS "Found Go compiler: ${GO_EXECUTABLE}")

set(LIB_NAME sha1cracker)
add_library(${LIB_NAME} SHARED cracker.cu)
target_compile_options(${LIB_NAME} PRIVATE -O3 -arch=native -Xcompiler -fPIC)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(EXECUTABLE_NAME sha1-crack)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/${EXECUTABLE_NAME}
    
    COMMAND ${CMAKE_COMMAND} -E env "CGO_LDFLAGS=-L${CMAKE_CURRENT_BINARY_DIR} -l${LIB_NAME}"
            ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/${EXECUTABLE_NAME} ${CMAKE_SOURCE_DIR}/main.go
    
    DEPENDS ${LIB_NAME}
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    
    COMMENT "Building Go executable..."
)

add_custom_target(${EXECUTABLE_NAME}_build_step ALL
    DEPENDS ${CMAKE_BINARY_DIR}/${EXECUTABLE_NAME}
)

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/${EXECUTABLE_NAME} DESTINATION bin PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)