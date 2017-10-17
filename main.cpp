#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorNURBSBasis.h"
#include "Topology.hpp"
#include "Surface.hpp"
#include "Vertex.hpp"
#include "DofMapper.hpp"
#include "PoissonMapper.hpp"

using namespace Eigen;
using namespace std;
using namespace Accessory;
using Coordinate=Element<2, 2, double>::Coordinate;
using CoordinatePairList=Element<2, 2, double>::CoordinatePairList;

using Vector1d = Matrix<double, 1, 1>;

int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0), point2(0, 3), point3(4, 0), point4(1, 1), point5(5, 0), point6(6, 4), point7(4,3);

    vector<Vector2d> point{point1, point2, point3, point4};
    vector<Vector2d> pointt{point3, point4, point5, point6};


    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, point);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, pointt);
    domain1->DegreeElevate(2);
    domain1->UniformRefine(2);

    Surface<2, double> cc(domain1, array<bool, 4>{false, false, false, false});
    auto surface1 = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, true, true, true});
    surface1->SurfaceInitialize();
    auto surface2 = make_shared<Surface<2, double>>(domain2, array<bool, 4>{false, false, false, false});
    surface2->SurfaceInitialize();
    surface1->PrintIndices(0);


    surface1->VertexPointerGetter(2)->PrintIndices(1);

    DofMapper<2,double> dof_map;
    PoissonMapper<2,double> mapper(dof_map);
    surface1->Accept(mapper);
    dof_map.PrintDirichletGlobalIndicesIn(domain1);
    return 0;
}