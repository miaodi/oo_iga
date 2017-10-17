cmake_minimum_required(VERSION 3.6)
project(oo_iga)

set(CMAKE_CXX_COMPILER /usr/local/bin/g++-7)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O2 -Wno-switch -fopenmp")
set(SOURCE_FILES main.cpp KnotVector.cpp KnotVector.h BsplineBasis.cpp BsplineBasis.h TensorBsplineBasis.cpp TensorBsplineBasis.h MultiArray.h PhyTensorBsplineBasis.h MmpMatrix.cpp MmpMatrix.h QuadratureRule.cpp QuadratureRule.h Topology.h Visitor.h DofMapper.h)
add_executable(oo_iga ${SOURCE_FILES})