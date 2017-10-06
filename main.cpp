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
    Vector2d point1(0, 1), point2(0, 2), point3(0, 3), point4(1, 1), point5(2, 2), point6(3, 3), point7(1, 0), point8(2,
                                                                                                                      0), point9(
            3, 0);
    vector<Vector2d> point{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(1), weight3(1), weight4(1.0 / sqrt(2)), weight5(1.0 / sqrt(2)), weight6(1.0 / sqrt(2)), weight7(1), weight8(
            1), weight9(1);
    vector<Vector1d> weight{weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, point, weight, true);


    Vector2d pt(.7, .8);
    Vector2d p = domain1->AffineMap(pt, vector<int>{0, 0});
    cout << p << endl;
    domain1->UniformRefine(1);
    p = domain1->AffineMap(pt, vector<int>{0, 0});
    cout << p << endl;
    cout<<setprecision(12)<<domain1->WtPtsGetter(4);
    domain1->PrintCtrPts();
    return 0;
}