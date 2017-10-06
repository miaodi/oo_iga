#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorNURBSBasis.h"
#include "Topology.h"


using namespace Eigen;
using namespace std;
using namespace Accessory;
using Coordinate=Element<double>::Coordinate;
using CoordinatePairList=Element<double>::CoordinatePairList;
using Quadrature = QuadratureRule<double>::Quadrature;
using QuadList = QuadratureRule<double>::QuadList;
using LoadFunctor = Element<double>::LoadFunctor;
using Vector1d = Matrix<double, 1, 1>;

int main() {
    KnotVector<double> a;
    a.InitClosed(2, 0, 1);
    Vector2d point1(0, 1), point2(1, 2), point3(0, 3);
    vector<Vector2d> point{point1, point2, point3};
    Vector1d weight1(1), weight2(1/sqrt(2)), weight3(1);
    vector<Vector1d> weight{weight1, weight2, weight3};
    auto domain1 = make_shared<PhyTensorNURBSBasis<1, 2, double>>(vector<KnotVector<double>>{a}, point, weight, true);
    Vector2d p;
    Vector1d pp,ppp;
    pp<<.2;
    cout<<ppp<<endl;
    domain1->InversePts(domain1->AffineMap(pp),ppp);
    cout<<ppp;
    return 0;
}