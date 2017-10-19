cmake_minimum_required(VERSION 3.6)
project(oo_iga)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Ofast -Wno-switch")

set(SOURCE_FILES main.cpp KnotVector.cpp KnotVector.h BsplineBasis.cpp BsplineBasis.h TensorBsplineBasis.cpp TensorBsplineBasis.h MultiArray.h PhyTensorBsplineBasis.h PhyTensorNURBSBasis.h Topology.hpp Surface.hpp Visitor.hpp Edge.hpp Vertex.hpp DofMapper.hpp PoissonMapper.hpp DomainVisitor.hpp QuadratureRule.h PoissonStiffnessVisitor.hpp)
add_executable(oo_iga ${SOURCE_FILES})