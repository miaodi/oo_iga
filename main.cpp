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
using Quadrature = QuadratureRule<double>::Quadrature ;
using QuadList = QuadratureRule<double>::QuadList;
int main() {

    QuadratureRule<double> hahaming(3);
    Vector2d begin(0, 0);
    Vector2d end(0, 2);
    pair<VectorXd,VectorXd> pai(begin,end);
    QuadList test;
    hahaming.MapToQuadrature(pai,test);
    for(auto &i:test)
        cout<<i.first.transpose()<<" weight: "<<i.second<<endl;
/*
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    KnotVector<double> b;
    b.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 2);
    Vector2d point3(1, .2);
    Vector2d point4(1, 1);


    Vector2d point5(-1, 1);
    Vector2d point6(0, 2);
    Vector2d point7(-2, -1);
    Vector2d point8(0, 0);
    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point5, point6, point7, point8});
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points2);
    domain1->DegreeElevate(0, 3);
    domain1->DegreeElevate(1, 6);
    domain1->UniformRefine(1, 4);
    domain1->UniformRefine(0, 1);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    cell1->Match(cell2);
    cell1->PrintEdgeInfo();
    cell2->PrintEdgeInfo();

    cout<<domain.AffineMap(Vector2d(.5,.2))<<endl;
    domain.DegreeElevate(1,2);
    domain.DegreeElevate(0,3);
    cout<<domain.AffineMap(Vector2d(0,1))<<endl;
    domain.UniformRefine(0,3);
    domain.UniformRefine(1,5);
    cout<<domain.AffineMap(Vector2d(.5,.2))<<endl;
    domain.PrintKnots(0);
    domain.PrintKnots(1);
    cout<<domain.Jacobian(Vector2d(.1,.2))<<endl;

    */
    return 0;
}