#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "Vertex.hpp"
#include "DofMapper.hpp"
#include "PoissonMapper.hpp"
#include "BiharmonicMapper.hpp"
#include "PoissonStiffnessVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "H2DomainNormVisitor.hpp"
#include "PoissonDirichletBoundaryVisitor.hpp"
#include "BiharmonicDirichletBoundaryVisitor.hpp"
#include "PoissonVertexVisitor.hpp"
#include "BiharmonicVertexVisitor.hpp"
#include "PoissonInterface.hpp"
#include "BiharmonicInterface.hpp"
#include "PostProcess.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;

template <typename T>
T operator^(T x, T y)
{
    return std::pow(x, y);
}

int main()
{
    double b = 1;
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(-1, b), point2(-1, b + 1), point3(0, 0), point4(0, 1), point5(1, 0), point6(1, 1);

    GeometryVector point{point1, point2, point3, point4};
    GeometryVector pointt{point3, point4, point5, point6};

    int degree, refine;
    cin >> degree >> refine;

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, point);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointt);

    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain1->KnotInsertion(1, 1.0 / 3);
    domain1->KnotInsertion(1, 2.0 / 3);
    domain1->UniformRefine(refine, 1);
    domain2->UniformRefine(refine + 1, 1);
    array<shared_ptr<Surface<2, double>>, 3> cells;
    cells[0] = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, false, true, true});
    cells[0]->SurfaceInitialize();
    cells[1] = make_shared<Surface<2, double>>(domain2, array<bool, 4>{true, true, true, false});
    cells[1]->SurfaceInitialize();

    for (int i = 0; i < 1; i++)
    {
        for (int j = i + 1; j < 2; j++)
            cells[i]->Match(cells[j]);
    }

    function<vector<double>(const VectorXd &)> body_force = [&b](const VectorXd &u) {
        double x = u(0);
        double y = u(1);

        return vector<double>{8 * (3 - 60 * y + 258 * pow(y, 2) - 340 * pow(y, 3) + 33 * pow(y, 4) + 156 * pow(y, 5) - 38 * pow(y, 6) - 12 * pow(y, 7) + 3 * pow(y, 8) +
                                   3 * pow(x, 4) * (1 - 20 * y + 90 * pow(y, 2) - 140 * pow(y, 3) + 70 * pow(y, 4)) +
                                   6 * pow(x, 2) * (-1 + 20 * y - 84 * pow(y, 2) + 100 * pow(y, 3) + 20 * pow(y, 4) - 84 * pow(y, 5) + 28 * pow(y, 6)) +
                                   6 * b * x * (-1 + 2 * y) * (5 - 29 * y - 13 * pow(y, 2) + 79 * pow(y, 3) - 27 * pow(y, 4) - 15 * pow(y, 5) + 5 * pow(y, 6) + 5 * pow(x, 4) * (1 - 7 * y + 7 * pow(y, 2)) + 10 * pow(x, 2) * (-1 + 6 * y + pow(y, 2) - 14 * pow(y, 3) + 7 * pow(y, 4))) +
                                   pow(b, 4) * (3 * pow(x, 8) + 3 * pow(-1 + y, 2) * pow(y, 2) + 2 * pow(x, 6) * (11 - 84 * y + 84 * pow(y, 2)) - 6 * pow(x, 2) * (-1 + 6 * y + 9 * pow(y, 2) - 30 * pow(y, 3) + 15 * pow(y, 4)) +
                                                3 * pow(x, 4) * (-9 + 60 * y + 10 * pow(y, 2) - 140 * pow(y, 3) + 70 * pow(y, 4))) +
                                   2 * pow(b, 3) * x * (15 * pow(x, 6) * (-1 + 2 * y) + 3 * pow(x, 4) * (3 + 64 * y - 210 * pow(y, 2) + 140 * pow(y, 3)) - 3 * (1 - 12 * y + 20 * pow(y, 2) + 20 * pow(y, 3) - 50 * pow(y, 4) + 20 * pow(y, 5)) + 5 * pow(x, 2) * (1 - 42 * y + 99 * pow(y, 2) + 4 * pow(y, 3) - 105 * pow(y, 4) + 42 * pow(y, 5))) +
                                   pow(b, 2) * (1 - 24 * y + 108 * pow(y, 2) - 132 * pow(y, 3) - 24 * pow(y, 4) + 108 * pow(y, 5) - 36 * pow(y, 6) + 3 * pow(x, 6) * (19 - 90 * y + 90 * pow(y, 2)) +
                                                9 * pow(x, 4) * (-11 + 20 * y + 130 * pow(y, 2) - 300 * pow(y, 3) + 150 * pow(y, 4)) +
                                                9 * pow(x, 2) * (5 + 2 * y - 117 * pow(y, 2) + 200 * pow(y, 3) - 25 * pow(y, 4) - 90 * pow(y, 5) + 30 * pow(y, 6))))};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [&b](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{pow(-1 + x, 2) * pow(1 + x, 2) * pow(-1 + y, 2) * pow(y, 2) * pow(-1 + b * x + y, 2) * pow(b * x + y, 2),
                              2 * (-1 + x) * (1 + x) * pow(-1 + y, 2) * pow(y, 2) * (-1 + b * x + y) * (b * x + y) * (pow(b, 2) * (-2 * x + 4 * pow(x, 3)) + 2 * x * (-1 + y) * y + b * (-1 + 3 * pow(x, 2)) * (-1 + 2 * y)),
                              2 * (-1 + x) * (1 + x) * pow(-1 + y, 2) * pow(y, 2) * (-1 + b * x + y) * (b * x + y) * (pow(b, 2) * (-2 * x + 4 * pow(x, 3)) + 2 * x * (-1 + y) * y + b * (-1 + 3 * pow(x, 2)) * (-1 + 2 * y))};
    };

    DofMapper<2, double> dof_map;
    BiharmonicMapper<2, double> mapper(dof_map);

    for (int i = 0; i < 2; i++)
    {
        cells[i]->Accept(mapper);
    }

    BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
    for (int i = 0; i < 2; i++)
    {
        cells[i]->Accept(stiffness);
    }
    SparseMatrix<double> stiffness_matrix_triangle_view, load_vector;
    stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
    stiffness.LoadAssembler(load_vector);
    SparseMatrix<double> stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();
    BiharmonicInterface<2, double> interface(dof_map);
    for (int i = 0; i < 2; i++)
    {
        cells[i]->EdgeAccept(interface);
    }
    MatrixXd constraint;
    interface.ConstraintMatrix(constraint);
    MatrixXd global_to_free = MatrixXd::Identity(dof_map.Dof(), dof_map.Dof());
    auto dirichlet_indices = dof_map.GlobalDirichletIndices();
    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); it++)
    {
        Accessory::removeRow(global_to_free, *it);
    }
    SparseMatrix<double> free_stiffness = SparseMatrix<double>(global_to_free.sparseView()) * stiffness_matrix * SparseMatrix<double>(global_to_free.transpose().sparseView());
    VectorXd free_load = global_to_free * load_vector;
    MatrixXd free_constraint = constraint * global_to_free.transpose();
    FullPivLU<MatrixXd> lu(free_constraint);
    lu.setThreshold(1e-10);
    MatrixXd ker = lu.kernel();
    SparseMatrix<double> ker_sparse = ker.sparseView();

    SparseMatrix<double> stiffness_sol = ker_sparse.transpose() * free_stiffness * ker_sparse;
    VectorXd load_sol = ker_sparse.transpose() * free_load;
    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(stiffness_sol);
    VectorXd Solution = cg.solve(load_sol);

    VectorXd solution = global_to_free.transpose() * ker_sparse * Solution;

    PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
    for (int i = 0; i < 2; i++)
    {
        cells[i]->Accept(post_process);
    }
    cout << post_process.L2Norm() << endl;

    vector<KnotVector<double>> solutionDomain1, solutionDomain2;
    solutionDomain1.push_back(domain1->KnotVectorGetter(0));
    solutionDomain1.push_back(domain1->KnotVectorGetter(1));
    solutionDomain2.push_back(domain2->KnotVectorGetter(0));
    solutionDomain2.push_back(domain2->KnotVectorGetter(1));
    VectorXd controlDomain1 = solution.segment(dof_map.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = solution.segment(dof_map.StartingIndex(domain2), domain2->GetDof());
    vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions(2);
    solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain1, controlDomain1);
    solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain2, controlDomain2);

    double x, y;
    ofstream file1, file2;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    for (int i = 0; i <= 100; i++)
    {
        for (int j = 0; j <= 100; j++)
        {
            double xi = 1.0 * i / 100, eta = 1.0 * j / 100;
            Vector2d u(xi, eta);
            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            auto result1 = (solutions[0]->AffineMap(u)(0) - analytical_solution(position1)[0]);
            auto result2 = (solutions[1]->AffineMap(u)(0) - analytical_solution(position2)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
        }
    }

    return 0;
}