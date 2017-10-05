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
    Vector2d point1(0, 0), point2(1, 0), point3(2, 0), point4(0, 1), point5(1, 1), point6(2, 1), point7(0, 2), point8(1,
                                                                                                                      2), point9(
            2, 2);
    vector<Vector2d> point{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(1), weight3(1), weight4(1), weight5(1), weight6(1), weight7(1), weight8(
            1), weight9(1);
    vector<Vector1d> weight{weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 2, double>>(vector<KnotVector<double>>{a,a}, point, weight, true);
    Vector2d pt(.1, .5);
    cout<< domain1->AffineMap(pt, vector<int>{0,0});
    cout<<domain1->Jacobian(pt);
    return 0;
}