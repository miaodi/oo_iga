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
    domain1->UniformRefine(1);
    domain1->PrintKnots(1);
    domain1->KnotInsertion(1,.6);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    cell1->PrintEdgeInfo();
    cell1->_edges[1]->PrintActivatedDofsOfLayers(0);


    DofMapper<double> s;
    PoissonMapperInitiator<double> visit(s);
    cell1->accept(visit);
    PoissonVisitor<double> poisson(s, [](Coordinate u)->vector<double>{
        return vector<double>({0});
    });
    cell1->accept(poisson);
    PoissonBoundaryVisitor<double> boundary(s, [](Coordinate u)->vector<double>{
        return vector<double>{sin(10*u(0))*sin(10*u(1))};
    });
    cell1->accept(boundary);
    poisson.StiffnessMatrix();
    boundary.Boundary();
    s.PrintDofIn(cell1->GetDomain());
    s.PrintFreeDofIn(cell1->GetDomain());
    s.PrintFreezedDofIn(cell1->GetDomain());
    auto indexmap = s.CondensedBiMap();
    auto transfer = BiMapToSparseMatrix<double>(s.FreeDof(),s.Dof(),indexmap);
    cout<< *transfer;
    return 0;
}