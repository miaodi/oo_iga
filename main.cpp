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

int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 4);
    Vector2d point2(0, 0);
    Vector2d point3(1, 3);
    Vector2d point4(1, 1);
    Vector2d point5(3, 3);
    Vector2d point6(3, 1);
    Vector2d point7(4, 4);
    Vector2d point8(4, 0);
    vector<Vector2d> points1({point1, point3, point2, point4});
    vector<Vector2d> points2({point2, point4, point8, point6});
    vector<Vector2d> points3({point4, point3, point6, point5});
    vector<Vector2d> points4({point6, point5, point8, point7});
    vector<Vector2d> points5({point3, point1, point5, point7});


    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points2);
    auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points3);
    auto domain4 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points4);
    auto domain5 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points5);
    domain1->DegreeElevate(5);
    domain2->DegreeElevate(5);
    domain3->DegreeElevate(5);
    domain4->DegreeElevate(5);
    domain5->DegreeElevate(5);

    domain1->UniformRefine(4);
    domain2->UniformRefine(4);
    domain3->UniformRefine(4);
    domain4->UniformRefine(4);
    domain5->UniformRefine(4);


    array<shared_ptr<Cell<double>>, 5> cells;
    cells[0] = make_shared<Cell<double>>(domain1);
    cells[1] = make_shared<Cell<double>>(domain2);
    cells[2] = make_shared<Cell<double>>(domain3);
    cells[3] = make_shared<Cell<double>>(domain4);
    cells[4] = make_shared<Cell<double>>(domain5);
    DofMapper<double> s;
    PoissonMapperInitiator<double> visit(s);
    for (int i = 0; i < 5; i++) {
        cells[i]->accept(visit);
    }
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++)
            cells[i]->Match(cells[j]);
    }
    const double pi = 3.141592653589793238462643383279502884;

    BiharmonicVisitor<double> biharmonic(s, [&pi](Coordinate u) -> vector<double> {
        return vector<double>{
                8 *
                (256 - 24 * pow(u(0), 3) + 3 * pow(u(0), 4) + 36 * pow(u(0) * (u(1) - 2), 2) - 384 * u(1) + 144 * pow(u(1), 2) - 24 * pow(u(1), 3) +
                 3 * pow(u(1), 4) - 48 * u(0) * (8 - 12 * u(1) + 3 * pow(u(1), 2)))};
    });

    for (int i = 0; i < 5; i++) {
        cells[i]->accept(biharmonic);
    }
    unique_ptr<SparseMatrix<double>> stiffness, load;
    tie(stiffness, load) = biharmonic.Domain();
    function<vector<double>(const Coordinate &)> Analytical = [&pi](const Coordinate &u) {
        return vector<double>{pow((u(0) - 4) * (u(1) - 4) * u(0) * u(1), 2), 4 * (u(0) - 4) * (u(0) - 2) * u(0) * pow(u(1) * (u(1) - 4), 2),
                              4 * (u(1) - 4) * (u(1) - 2) * u(1) * pow(u(0) * (u(0) - 4), 2)};
    };

    BiharmonicDGBoundaryVisitor<double> boundary(s, Analytical);

    for (int i = 0; i < 5; i++) {
        cells[i]->accept(boundary);
    }

    unique_ptr<SparseMatrix<double>> boundaryStiffness, boundaryLoad;
    tie(boundaryStiffness, boundaryLoad) = boundary.Boundary();

    BiharmonicDGInterfaceVisitor<double> interface(s);

    for (int i = 0; i < 5; i++) {
        cells[i]->accept(interface);
    }

    auto interfaceStiffness = interface.DGInterface();
    SparseMatrix<double> stiffnessSol = *stiffness + *interfaceStiffness + *boundaryStiffness;
    VectorXd loadSol = *load + *boundaryLoad;
    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.setMaxIterations(1e8);
    cg.compute(stiffnessSol);
    VectorXd Solution = cg.solve(loadSol);
    cout<<Solution<<endl;
    vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3, solutionDomain4, solutionDomain5;
    solutionDomain1.push_back(domain1->KnotVectorGetter(0));
    solutionDomain1.push_back(domain1->KnotVectorGetter(1));
    solutionDomain2.push_back(domain2->KnotVectorGetter(0));
    solutionDomain2.push_back(domain2->KnotVectorGetter(1));
    solutionDomain3.push_back(domain3->KnotVectorGetter(0));
    solutionDomain3.push_back(domain3->KnotVectorGetter(1));
    solutionDomain4.push_back(domain4->KnotVectorGetter(0));
    solutionDomain4.push_back(domain4->KnotVectorGetter(1));
    solutionDomain5.push_back(domain5->KnotVectorGetter(0));
    solutionDomain5.push_back(domain5->KnotVectorGetter(1));
    VectorXd controlDomain1 = Solution.segment(s.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = Solution.segment(s.StartingIndex(domain2), domain2->GetDof());
    VectorXd controlDomain3 = Solution.segment(s.StartingIndex(domain3), domain3->GetDof());
    VectorXd controlDomain4 = Solution.segment(s.StartingIndex(domain4), domain4->GetDof());
    VectorXd controlDomain5 = Solution.segment(s.StartingIndex(domain5), domain5->GetDof());
    auto solution1 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain1, controlDomain1);
    auto solution2 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain2, controlDomain2);
    auto solution3 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain3, controlDomain3);
    auto solution4 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain4, controlDomain4);
    auto solution5 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain5, controlDomain5);
    double x, y;

    ofstream file1, file2, file3, file4, file5;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    file3.open("domain3.txt");
    file4.open("domain4.txt");
    file5.open("domain5.txt");
    for (int i = 0; i <= 50; i++) {
        for (int j = 0; j <= 50; j++) {
            double xi = 1.0 * i / 50, eta = 1.0 * j / 50;
            Vector2d u(xi, eta);

            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            VectorXd position3 = domain3->AffineMap(u);
            VectorXd position4 = domain4->AffineMap(u);
            VectorXd position5 = domain5->AffineMap(u);
            auto result1 = abs(solution1.AffineMap(u)(0) - Analytical(position1)[0]);
            auto result2 = abs(solution2.AffineMap(u)(0) - Analytical(position2)[0]);
            auto result3 = abs(solution3.AffineMap(u)(0) - Analytical(position3)[0]);
            auto result4 = abs(solution4.AffineMap(u)(0) - Analytical(position4)[0]);
            auto result5 = abs(solution5.AffineMap(u)(0) - Analytical(position5)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
            file3 << position3(0) << " " << position3(1) << " " << result3 << endl;
            file4 << position4(0) << " " << position4(1) << " " << result4 << endl;
            file5 << position5(0) << " " << position5(1) << " " << result5 << endl;
        }
    }


    /*
    auto interfaceStiffness = interface.DGInterface();
    SparseMatrix<double> stiffnessSol = *stiffness + *interfaceStiffness + *boundaryStiffness;
    VectorXd loadSol = *load + *boundaryLoad;
    SparseLU<SparseMatrix<double> > solver;
    solver.compute(stiffnessSol);
    VectorXd Solution =solver.solve(loadSol);
    vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3, solutionDomain4, solutionDomain5;
    solutionDomain1.push_back(domain1->KnotVectorGetter(0));
    solutionDomain1.push_back(domain1->KnotVectorGetter(1));
    solutionDomain2.push_back(domain2->KnotVectorGetter(0));
    solutionDomain2.push_back(domain2->KnotVectorGetter(1));
    solutionDomain3.push_back(domain3->KnotVectorGetter(0));
    solutionDomain3.push_back(domain3->KnotVectorGetter(1));
    solutionDomain4.push_back(domain4->KnotVectorGetter(0));
    solutionDomain4.push_back(domain4->KnotVectorGetter(1));
    solutionDomain5.push_back(domain5->KnotVectorGetter(0));
    solutionDomain5.push_back(domain5->KnotVectorGetter(1));
    VectorXd controlDomain1 = Solution.segment(s.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = Solution.segment(s.StartingIndex(domain2), domain2->GetDof());
    VectorXd controlDomain3 = Solution.segment(s.StartingIndex(domain3), domain3->GetDof());
    VectorXd controlDomain4 = Solution.segment(s.StartingIndex(domain4), domain4->GetDof());
    VectorXd controlDomain5 = Solution.segment(s.StartingIndex(domain5), domain5->GetDof());
    auto solution1 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain1, controlDomain1);
    auto solution2 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain2, controlDomain2);
    auto solution3 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain3, controlDomain3);
    auto solution4 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain4, controlDomain4);
    auto solution5 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain5, controlDomain5);
    double x, y;

    ofstream file1, file2, file3, file4, file5;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    file3.open("domain3.txt");
    file4.open("domain4.txt");
    file5.open("domain5.txt");
    for (int i = 0; i <= 100; i++) {
        for (int j = 0; j <= 100; j++) {
            double xi = 1.0 * i / 100, eta = 1.0 * j / 100;
            Vector2d u(xi, eta);

            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            VectorXd position3 = domain3->AffineMap(u);
            VectorXd position4 = domain4->AffineMap(u);
            VectorXd position5 = domain5->AffineMap(u);
            auto result1 = abs(solution1.AffineMap(u)(0) - Analytical(position1)[0]);
            auto result2 = abs(solution2.AffineMap(u)(0) - Analytical(position2)[0]);
            auto result3 = abs(solution3.AffineMap(u)(0) - Analytical(position3)[0]);
            auto result4 = abs(solution4.AffineMap(u)(0) - Analytical(position4)[0]);
            auto result5 = abs(solution5.AffineMap(u)(0) - Analytical(position5)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
            file3 << position3(0) << " " << position3(1) << " " << result3 << endl;
            file4 << position4(0) << " " << position4(1) << " " << result4 << endl;
            file5 << position5(0) << " " << position5(1) << " " << result5 << endl;
        }
    }

    return 0;
    /*
    const double pi = 3.141592653589793238462643383279502884;

    BiharmonicVisitor<double> biharmonic(s, [&pi](Coordinate u) -> vector<double> {
        return vector<double>{4 * pow(pi, 4) * sin(pi * u(0)) * sin(pi * u(1))};
    });

    for (int i = 0; i < 5; i++) {
        cells[i]->accept(biharmonic);
    }


    function<vector<double>(const Coordinate &)> Analytical = [&pi](const Coordinate &u) {
        return vector<double>{sin(pi * u(0)) * sin(pi * u(1)), pi * cos(pi * u(0)) * sin(pi * u(1)),
                              pi * sin(pi * u(0)) * cos(pi * u(1))};
    };
    BiharmonicBoundaryVisitor<double> boundary(s, Analytical);

    for (int i = 0; i < 5; i++) {
        cells[i]->accept(boundary);
    }

    s.PrintSlaveDofIn(domain2);
    s.PrintDofIn(domain1);
    BiharmonicInterfaceVisitor<double> interface(s);

    for (int i = 0; i < 5; i++) {
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
    solutionDomain3.push_back(domain3->KnotVectorGetter(0));
    solutionDomain3.push_back(domain3->KnotVectorGetter(1));
    solutionDomain4.push_back(domain4->KnotVectorGetter(0));
    solutionDomain4.push_back(domain4->KnotVectorGetter(1));
    solutionDomain5.push_back(domain5->KnotVectorGetter(0));
    solutionDomain5.push_back(domain5->KnotVectorGetter(1));
    VectorXd controlDomain1 = solution.segment(s.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = solution.segment(s.StartingIndex(domain2), domain2->GetDof());
    VectorXd controlDomain3 = solution.segment(s.StartingIndex(domain3), domain3->GetDof());
    VectorXd controlDomain4 = solution.segment(s.StartingIndex(domain4), domain4->GetDof());
    VectorXd controlDomain5 = solution.segment(s.StartingIndex(domain5), domain5->GetDof());
    auto solution1 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain1, controlDomain1);
    auto solution2 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain2, controlDomain2);
    auto solution3 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain3, controlDomain3);
    auto solution4 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain4, controlDomain4);
    auto solution5 = PhyTensorBsplineBasis<2, 1, double>(solutionDomain5, controlDomain5);
    double x, y;

    ofstream file1, file2, file3, file4, file5;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    file3.open("domain3.txt");
    file4.open("domain4.txt");
    file5.open("domain5.txt");
    for (int i = 0; i <= 300; i++) {
        for (int j = 0; j <= 300; j++) {
            double xi = 1.0 * i / 300, eta = 1.0 * j / 300;
            Vector2d u(xi, eta);

            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            VectorXd position3 = domain3->AffineMap(u);
            VectorXd position4 = domain4->AffineMap(u);
            VectorXd position5 = domain5->AffineMap(u);
            auto result1 = abs(solution1.AffineMap(u)(0) - Analytical(position1)[0]);
            auto result2 = abs(solution2.AffineMap(u)(0) - Analytical(position2)[0]);
            auto result3 = abs(solution3.AffineMap(u)(0) - Analytical(position3)[0]);
            auto result4 = abs(solution4.AffineMap(u)(0) - Analytical(position4)[0]);
            auto result5 = abs(solution5.AffineMap(u)(0) - Analytical(position5)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
            file3 << position3(0) << " " << position3(1) << " " << result3 << endl;
            file4 << position4(0) << " " << position4(1) << " " << result4 << endl;
            file5 << position5(0) << " " << position5(1) << " " << result5 << endl;
        }
    }
    time(&end);
    std::cout << difftime(end, start) << " seconds" << std::endl;
    */
    return 0;
}