#!/bin/bash

git clone https://github.com/openxla/stablehlo -b v1.8.6

mkdir -p build && cd build
# Set the LLVM_ENABLE_LLD shell variable depending on your preferences. 
# We recommend setting it to ON on Linux and to OFF on macOS.

export LLVM_DIR=/usr/local/lib/cmake/llvm
export PATH=/usr/local/bin:$PATH

CC=/usr/bin/clang-18 CXX=/usr/bin/clang++-18 \
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir

cmake --build .



