#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorBsplineBasis.h"
#include "QuadratureRule.h"
#include "Topology.h"
#include <fstream>
#include <iomanip>
#include <ctime>

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
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 2);
    Vector2d point3(1, 0);
    Vector2d point4(1, 2);
    Vector2d point5(2, 0);
    Vector2d point6(2, 2);

    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point3, point4, point5, point6});
    vector<Vector2d> points3({point1, point2});

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points2);
    domain1->DegreeElevate(3);
    domain2->DegreeElevate(3);

    domain1->UniformRefine(2);
    domain2->UniformRefine(2);

    vector<shared_ptr<Cell<double>>> cells(2);
    cells[0] = make_shared<Cell<double>>(domain1);
    cells[1] = make_shared<Cell<double>>(domain2);
    for (int i = 0; i < 1; i++) {
        for (int j = i + 1; j < 2; j++)
            cells[i]->Match(cells[j]);
    }
    for (int i = 0; i < 2; i++) {
        cells[i]->PrintEdgeInfo();
    }


    DofMapper<double> s;
    BiharmonicMapperInitiator<double> visit(s);
    for (int i = 0; i < 2; i++) {
        cells[i]->accept(visit);
    }

    const double pi = 3.141592653589793238462643383279502884;

    BiharmonicVisitor<double> biharmonic(s, [&pi](Coordinate u) -> vector<double> {
        return vector<double>{4 * pow(pi, 4) * sin(pi * u(0)) * sin(pi * u(1))};
    });

    for (int i = 0; i < 2; i++) {
        cells[i]->accept(biharmonic);
    }

    function<vector<double>(const Coordinate &)> Analytical = [&pi](const Coordinate &u) {
        return vector<double>{sin(pi * u(0)) * sin(pi * u(1)), pi * cos(pi * u(0)) * sin(pi * u(1)),
                              pi * sin(pi * u(0)) * cos(pi * u(1))};
    };
    BiharmonicBoundaryVisitor<double> boundary(s, Analytical);

    for (int i = 0; i < 2; i++) {
        cells[i]->accept(boundary);
    }

    BiharmonicInterfaceVisitor<double> interface(s);

    for (int i = 0; i < 2; i++) {
        cells[i]->accept(interface);
    }
    unique_ptr<SparseMatrix<double>> coupling = interface.Coupling();

    unique_ptr<SparseMatrix<double>> stiffness, load, boundaryValue;
    tie(stiffness, load) = biharmonic.Domain();
    boundaryValue = boundary.Boundary();

    *stiffness = *coupling * (*stiffness) * coupling->transpose();
    VectorXd loadSum = (*coupling * *load) - (*stiffness * *boundaryValue);
    auto freedof = s.CondensedIndexMap();

    VectorXd loadSol = *SparseTransform<double>(freedof, s.Dof()) * loadSum;
    unique_ptr<SparseMatrix<double>> stiffnessSol = SparseMatrixGivenColRow<double>(freedof, freedof, stiffness);

    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(*stiffnessSol);
    VectorXd Solution = cg.solve(loadSol);
    VectorXd boundaryDense = VectorXd(*boundaryValue);
    VectorXd solution = coupling->transpose() * (SparseTransform<double>(freedof, s.Dof())->transpose() * Solution + boundaryDense);

    vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3, solutionDomain4, solutionDomain5;
    solutionDomain1.push_back(domain1->KnotVectorGetter(0));
    solutionDomain1.push_back(domain1->KnotVectorGetter(1));
    solutionDomain2.push_back(domain2->KnotVectorGetter(0));
    solutionDomain2.push_back(domain2->KnotVectorGetter(1));
    VectorXd controlDomain1 = solution.segment(s.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = solution.segment(s.StartingIndex(domain2), domain2->GetDof());
    auto solution1 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain1, controlDomain1);
    auto solution2 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain2, controlDomain2);
    double x, y;

    ofstream file1, file2;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    for (int i = 0; i <= 100; i++) {
        for (int j = 0; j <= 100; j++) {
            double xi = 1.0 * i / 100, eta = 1.0 * j / 100;
            Vector2d u(xi, eta);
            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            auto result1 = abs(solution1.AffineMap(u)(0) - Analytical(position1)[0]);
            auto result2 = abs(solution2.AffineMap(u)(0) - Analytical(position2)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
        }
    }
    return 0;
}