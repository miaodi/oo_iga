cmake_minimum_required(VERSION 3.6)
project(oo_iga)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Ofast -Wno-switch -fopenmp -march=native")
set(CMAKE_EXE_LINKER_FLAGS "-lgmp")


set(SPECTRA_INCLUDE_DIR $ENV{HOME}/spectra/include)
include_directories(${SPECTRA_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${arpack_ng_INCLUDE_DIRS})
LINK_DIRECTORIES(${arpack_ng_LIBRARIES})

include_directories(include)
file(GLOB SOURCE_FILES "src/*.cpp")
add_executable(oo_iga ${SOURCE_FILES})
# add_executable(oo_iga   src/BsplineBasis.cpp
#                         src/KnotVector.cpp
#                         src/Utility.cpp
#                         src/main.cpp
#                         src/QuadratureRule.cpp)
target_link_libraries(oo_iga gmp arpack)



#  MultiArray.h PhyTensorBsplineBasis.h PhyTensorNURBSBasis.h Topology.hpp Surface.hpp Visitor.hpp Edge.hpp Vertex.hpp DofMapper.hpp PoissonMapper.hpp DomainVisitor.hpp PoissonStiffnessVisitor.hpp DirichletBoundaryVisitor.hpp StiffnessVisitor.hpp BiharmonicStiffnessVisitor.hpp PoissonDirichletBoundaryVisitor.hpp AbstractMapper.hpp BiharmonicMapper.hpp BiharmonicDirichletBoundaryVisitor.hpp InterfaceVisitor.hpp Utility.hpp PoissonInterface.hpp PostProcess.hpp PoissonVertexVisitor.hpp
#-DEIGEN_USE_BLAS