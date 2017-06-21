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

    domain1->DegreeElevate(1);
    domain2->DegreeElevate(1);
    domain1->UniformRefine(1);
    domain1->KnotInsertion(0,.32);
    domain1->KnotInsertion(1,.62);

    domain2->PrintKnots(1);
    shared_ptr<Cell<double>> cell1 = make_shared<Cell<double>>(domain1);
    shared_ptr<Cell<double>> cell2 = make_shared<Cell<double>>(domain2);
    cell1->Match(cell2);
    cell1->PrintEdgeInfo();
    cell2->PrintEdgeInfo();
    cell1->_edges[1]->PrintActivatedDofsOfLayers(0);
    auto slaveIndex = cell1->_edges[1]->AllActivatedDofsOfLayersExcept(0,1);
    DofMapper<double> s;
    PoissonMapperInitiator<double> visit(s);
    cell1->accept(visit);
    cell2->accept(visit);
    PoissonVisitor<double> poisson(s, [](Coordinate u) -> vector<double> {
        return vector<double>{sin(u(0)*u(1))*u(1)*u(1)+sin(u(0)*u(1))*u(0)*u(0)};
    });
    cell1->accept(poisson);
    function<vector<double>(const Coordinate &)> Analytical =[](const Coordinate &u){return vector<double>{sin(u(0)*u(1))+u(0)*u(1)};};
    PoissonBoundaryVisitor<double> boundary(s, Analytical);
    cell1->accept(boundary);
    s.PrintSlaveDofIn(domain2);
    PoissonInterfaceVisitor<double> interface(s);
    cell1->accept(interface);
    cell2->accept(interface);
    /*
unique_ptr<SparseMatrix<double>> stiffness, load, boundaryValue;
tie(stiffness, load) = poisson.Domain();
boundaryValue = boundary.Boundary();

auto freedof = s.CondensedIndexMap();
SparseMatrix<double> loadSum = (*load - *stiffness * (*boundaryValue));
auto loadSol = SparseMatrixGivenColRow<double>(freedof, vector<int>{0}, loadSum);
auto stiffnessSol = SparseMatrixGivenColRow<double>(freedof, freedof, stiffness);
ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
cg.compute(*stiffnessSol);
MatrixXd solution = cg.solve(*loadSol);
solution=SparseTransform<double>(freedof,s.Dof())->transpose()*solution+MatrixXd(*boundaryValue);
vector<KnotVector<double>> solutionKnot;
solutionKnot.push_back(domain1->KnotVectorGetter(0));
solutionKnot.push_back(domain1->KnotVectorGetter(1));
auto solutionDomain = PhyTensorBsplineBasis<2, 1, double>(solutionKnot,solution);
Vector2d u(.145, .73);

cout<<solutionDomain.AffineMap(u)<<endl;
cout<<Analytical(domain1->AffineMap(u))[0];
*/
    return 0;
}