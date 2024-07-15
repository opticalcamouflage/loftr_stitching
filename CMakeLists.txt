cmake_minimum_required(VERSION 3.10)

project(demo VERSION 0.0.2)
set (CMAKE_CXX_STANDARD 17)

add_executable(demo ./LoFTR.cpp
                    ./LoFTRApp.cpp
                    ./OrtSessionHandler.cpp
)

set(ORT_ROOT "C:/Program Files/onnxruntime-win-x64-gpu-1.18.0")

target_include_directories(demo
    PRIVATE
    "${ORT_ROOT}/include"
    "${ORT_ROOT}/include/onnxruntime/core/session"
    ./
    ./ort_utility/
)

target_link_directories(demo
    PRIVATE
    "${ORT_ROOT}/lib"
    "${ORT_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/shared"
)

target_link_libraries(demo PRIVATE onnxruntime
                                    onnxruntime_providers_cuda
                                    onnxruntime_providers_shared
                                    onnxruntime_providers_tensorrt)

# OpenCV
set(OpenCV_STATIC OFF)
set(OpenCV_DIR "C:/opencv/opencv/build")
find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(demo PRIVATE ${OpenCV_LIBS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

find_package(CUDAToolkit)