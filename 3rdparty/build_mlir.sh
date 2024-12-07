#!/bin/bash

# https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md

git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout tags/llvmorg-18.1.8 && cd ..

mkdir llvm-project/build
cd llvm-project/build

wget https://github.com/Kitware/CMake/releases/download/v3.30.6/cmake-3.30.6-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.30.6-linux-x86_64.tar.gz && \
    cd cmake-3.30.6-linux-x86_64 && \
    cp -r * /usr && \
    cd .. && \
    rm -rf cmake-3.30.6-linux-x86_64 cmake-3.30.6-linux-x86_64.tar.gz

CC=/usr/bin/clang-18 CXX=/usr/bin/clang++-18 \
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_LLD=ON
   
ninja -j$(nproc)
# cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/mlir-tutorial/install
cmake --build . --target check-mlir
