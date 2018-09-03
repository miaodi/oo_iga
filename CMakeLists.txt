cmake_minimum_required(VERSION 3.6)
project(oo_iga)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Ofast -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "-lgmp")

include_directories(include)
file(GLOB SOURCE_FILES "src/*.cpp")
add_executable(oo_iga ${SOURCE_FILES})
target_link_libraries(oo_iga gmp )



#  MultiArray.h PhyTensorBsplineBasis.h PhyTensorNURBSBasis.h Topology.hpp Surface.hpp Visitor.hpp Edge.hpp Vertex.hpp DofMapper.hpp PoissonMapper.hpp DomainVisitor.hpp PoissonStiffnessVisitor.hpp DirichletBoundaryVisitor.hpp StiffnessVisitor.hpp BiharmonicStiffnessVisitor.hpp PoissonDirichletBoundaryVisitor.hpp AbstractMapper.hpp BiharmonicMapper.hpp BiharmonicDirichletBoundaryVisitor.hpp InterfaceVisitor.hpp Utility.hpp PoissonInterface.hpp PostProcess.hpp PoissonVertexVisitor.hpp
#-DEIGEN_USE_BLAS