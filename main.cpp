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
    a.InitClosed(2, 0, 1);
    a.Insert(1.0/3);
    a.Insert(1.0/3);
    a.Insert(2.0/3);
    a.printKnotVector();
    auto result = BezierExtraction(a,2);
    for(auto i:*result){
        cout<<i<<endl<<endl;
    }
    BsplineBasis<double> c(a);

    MatrixXd graminv = GramianInverse<double>(8);
    MatrixXd gram = Gramian<double>(8);
    cout<<gram*graminv<<endl;
    auto res = AllBernstein<double>(3,-1);
    for(auto i:res)
        cout<<i<<" ";
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