#include <iostream>
#include<eigen3/Eigen/Dense>
#include "PhyTensorBsplineBasis.h"
#include "QuadratureRule.h"
#include "Topology.h"
#include "DofMapper.h"

using namespace Eigen;
using namespace std;
using namespace Accessory;
using Coordinate=Element<double>::Coordinate;
using CoordinatePairList=Element<double>::CoordinatePairList;
using Quadrature = QuadratureRule<double>::Quadrature;
using QuadList = QuadratureRule<double>::QuadList;
using LoadFunctor = Element<double>::LoadFunctor;
int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    KnotVector<double> b;
    b.InitClosed(1, -1, 3);
    Vector2d point1(0, 4);
    Vector2d point2(2, 4);
    Vector2d point3(0, 0);
    Vector2d point4(2, 2);


    Vector2d point5(0, 0);
    Vector2d point6(2, 2);
    Vector2d point7(4, 0);
    Vector2d point8(4, 2);
    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point5, point6, point7, point8});
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(b, b, points2);
    domain1->DegreeElevate(2);
    domain2->DegreeElevate(2);
    domain1->UniformRefine(5);
    domain2->UniformRefine(5);
    domain1->KnotInsertion(1,.6,1);
    Vector2d u(.2, .6);
    domain1->PrintKnots(1);
    cout<<domain1->AffineMap(u);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    cell1->Match(cell2);
    cell1->PrintEdgeInfo();
    cell2->PrintEdgeInfo();
    DofMapper<double> s;
    PoissonMapperInitiator<double> visit(s);
    cell1->accept(visit);
    cell2->accept(visit);
    cout<<s.Dof()<<endl;
    PoissonVisitor<double> poisson(s, [](Coordinate u)->vector<double>{
        return vector<double>({0});
    });
    cell1->accept(poisson);
    cell2->accept(poisson);
    PoissonBoundaryVisitor<double> boundary(s, [](Coordinate u)->vector<double>{
        return vector<double>{sin(u(0))*sin(u(1))};
    });
    cell1->accept(boundary);
    cell2->accept(boundary);
    poisson.StiffnessMatrix();
    boundary.Boundary();
/*
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    auto indices = domain1->AllActivatedDofsOnBoundary(1,0);
    LoadFunctor haha;
    haha = [] (const VectorXd & u){ return vector<double> {pow(u(0)*u(1),1)};};
    PoissonDomainVisitor<double> poisson;
    cell1->accept(poisson, haha);
    auto matrix = poisson.MakeDenseVector();
    cout<<*matrix;

    PoissonBoundaryVisitor<double> poissonBoundary;
    cell1->_edges[2]->accept(poissonBoundary, haha);

    auto matrix1 = poissonBoundary.MakeDenseMatrix();
    cout<<*matrix1<<endl;
    mmpMatrix<double> edgemass(*poissonBoundary.MakeDenseMatrix());
    edgemass.removeZero();
    mmpMatrix<double> edgeload(*poissonBoundary.MakeDenseVector());
    edgeload.removeZero();
    edgemass.triangularView<Eigen::Lower>().solveInPlace(edgeload);
    cout<<edgeload<<endl;
*/
    return 0;
}