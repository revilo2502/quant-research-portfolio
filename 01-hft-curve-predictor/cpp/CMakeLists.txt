cmake_minimum_required(VERSION 3.15)
project(hft_curve_predictor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# --- order book replay engine: always built, zero external deps ---
add_executable(replay_engine orderbook/replay_engine.cpp)
target_include_directories(replay_engine PRIVATE orderbook)

# --- ONNX inference binary: only built if ONNX Runtime is found ---
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime
        $ENV{ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS /usr/local/lib /usr/lib
        $ENV{ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
  message(STATUS "ONNX Runtime found -- building infer_onnx")
  add_executable(infer_onnx inference/infer_onnx.cpp)
  target_include_directories(infer_onnx PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(infer_onnx PRIVATE ${ONNXRUNTIME_LIB})
else()
  message(STATUS "ONNX Runtime NOT found -- skipping infer_onnx "
                  "(see README.md 'Optional: C++ ONNX inference' to enable it)")
endif()
