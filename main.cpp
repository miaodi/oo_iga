#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorNURBSBasis.h"
#include "Topology.hpp"
#include "Surface.hpp"
#include "Vertex.hpp"

using namespace Eigen;
using namespace std;
using namespace Accessory;
using Coordinate=Element<2, 2, double>::Coordinate;
using CoordinatePairList=Element<2, 2, double>::CoordinatePairList;

using Vector1d = Matrix<double, 1, 1>;

int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0), point2(0, 3), point3(4, 0), point4(1, 1);
    vector<Vector2d> point{point1, point2, point3, point4};
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, point);
    auto edge1 = make_shared<PhyTensorBsplineBasis<1, 2, double>>(vector<KnotVector<double>>{a}, vector<Vector2d>{point1,point2});
    auto vertex1 = make_shared<PhyTensorBsplineBasis<0, 2, double>>(point1);
    Surface<2,double> surface(domain1);
    auto edge=surface.EdgePointerGetter(0);
    auto domain=edge->GetDomain();
    cout<<domain->AffineMap(Vector1d(.6));
    surface.PrintEdgeInfo();
    return 0;
}