cmake_minimum_required(VERSION 3.6)
project(oo_iga)

set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
set(CMAKE_CXX_STANDARD 14)
set(CLANG_LINK_DIR /usr/local/opt/llvm/lib)
set(CMAKE_CXX_FLAGS "-Ofast -Wno-switch -fopenmp -march=native")
set(CMAKE_EXE_LINKER_FLAGS "-lgmp")
link_directories(${CLANG_LINK_DIR} ${OPENBLAS_LINK_DIR})
include_directories(${OPENBLAS_INCLUDE})

set(SPECTRA_INCLUDE_DIR $ENV{HOME}/Spectra/include/Spectra)
include_directories(${SPECTRA_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${OPENBLAS_INCLUDE})
include_directories(include)
file(GLOB SOURCE_FILES "src/*.cpp")
add_executable(oo_iga ${SOURCE_FILES})



#  MultiArray.h PhyTensorBsplineBasis.h PhyTensorNURBSBasis.h Topology.hpp Surface.hpp Visitor.hpp Edge.hpp Vertex.hpp DofMapper.hpp PoissonMapper.hpp DomainVisitor.hpp PoissonStiffnessVisitor.hpp DirichletBoundaryVisitor.hpp StiffnessVisitor.hpp BiharmonicStiffnessVisitor.hpp PoissonDirichletBoundaryVisitor.hpp AbstractMapper.hpp BiharmonicMapper.hpp BiharmonicDirichletBoundaryVisitor.hpp InterfaceVisitor.hpp Utility.hpp PoissonInterface.hpp PostProcess.hpp PoissonVertexVisitor.hpp
#-DEIGEN_USE_BLAS