cmake_minimum_required(VERSION 3.6)
project(oo_iga)

# set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++-7)
set(CMAKE_CXX_STANDARD 14)
# set(CLANG_LINK_DIR /usr/local/opt/llvm/lib)
set(OPENBLAS_LINK_DIR /usr/local/opt/openblas/lib)
set(OPENBLAS_INCLUDE /usr/local/opt/openblas/include)
set(CMAKE_CXX_FLAGS "-Ofast -Wno-switch -fopenmp -DEIGEN_USE_BLAS -march=native")
set(CMAKE_EXE_LINKER_FLAGS "-lopenblas -lgmp")

link_directories(${CLANG_LINK_DIR} ${OPENBLAS_LINK_DIR})
include_directories(${OPENBLAS_INCLUDE})
include_directories(include)
file(GLOB SOURCE_FILES "src/*.cpp")
add_executable(oo_iga ${SOURCE_FILES})


#  MultiArray.h PhyTensorBsplineBasis.h PhyTensorNURBSBasis.h Topology.hpp Surface.hpp Visitor.hpp Edge.hpp Vertex.hpp DofMapper.hpp PoissonMapper.hpp DomainVisitor.hpp PoissonStiffnessVisitor.hpp DirichletBoundaryVisitor.hpp StiffnessVisitor.hpp BiharmonicStiffnessVisitor.hpp PoissonDirichletBoundaryVisitor.hpp AbstractMapper.hpp BiharmonicMapper.hpp BiharmonicDirichletBoundaryVisitor.hpp InterfaceVisitor.hpp Utility.hpp PoissonInterface.hpp PostProcess.hpp PoissonVertexVisitor.hpp