#include <iostream>
#include<eigen3/Eigen/Dense>
#include "PhyTensorBsplineBasis.h"
#include "MultiArray.h"
#include "MmpMatrix.h"
#include "QuadratureRule.h"
#include "Topology.h"

using namespace Eigen;
using namespace std;
using namespace Accessory;
using CoordinatePairList=Element<double>::CoordinatePairList;
using Quadrature = QuadratureRule<double>::Quadrature;
using QuadList = QuadratureRule<double>::QuadList;
using LoadFunctor = Visitor<double>::LoadFunctor;
int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    KnotVector<double> b;
    b.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 1);
    Vector2d point3(1, 0);
    Vector2d point4(1, 1);


    Vector2d point5(-1, 1);
    Vector2d point6(0, 2);
    Vector2d point7(-2, -1);
    Vector2d point8(0, 0);
    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point5, point6, point7, point8});
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points2);
    domain1->UniformRefine(1,3);
    domain1->UniformRefine(0,2);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    auto indices = domain1->AllActivatedDofsOnBoundary(1,0);
    for(auto &i:*indices)
        cout<<i<<" ";
/*
    PoissonVisitor<double> poisson;
    cell1->accept(poisson);
*/

    return 0;
}