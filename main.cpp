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
    Vector2d point1(0, 4);
    Vector2d point2(2, 4);
    Vector2d point3(0, 0);
    Vector2d point4(2, 2);


    Vector2d point5(-1, 1);
    Vector2d point6(0, 2);
    Vector2d point7(-2, -1);
    Vector2d point8(0.887298, 0.887298);
    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point5, point6, point7, point8});
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points2);
    domain1->DegreeElevate(0,1);
    domain1->DegreeElevate(1,1);
    domain1->UniformRefine(0,5);
    domain1->UniformRefine(1,5);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    auto indices = domain1->AllActivatedDofsOnBoundary(1,0);
    LoadFunctor haha;
    haha = [] (const VectorXd & u){ return vector<double> {u(0)*u(1)};};
    PoissonVisitor<double> poisson;
    cell1->accept(poisson, haha);
    auto matrix = poisson.MakeDenseVector();
    cout<<*matrix;

    return 0;
}