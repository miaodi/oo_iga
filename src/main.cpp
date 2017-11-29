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
#include "BiharmonicInterfaceH1.h"
#include "PostProcess.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;

int main()
{
    double b = 1.0;
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 1), point2(.5, (1 + b) / 2.0), point3(1, b), point4(0, 0), point5(.4, .57), point6(1, .5), point7(1, 0);

    GeometryVector point{point1, point2, point4, point5};
    GeometryVector pointt{point2, point3, point5, point6};
    GeometryVector pointtt{point4, point5, point7, point6};

    int degree, refine;
    cin >> degree >> refine;
    for (int i = 1; i < degree; i++)
    {
        for (int j = 0; j < refine; j++)
        {

            auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, point);
            auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointt);
            auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointtt);

            domain1->DegreeElevate(i);
            domain2->DegreeElevate(i);
            domain3->DegreeElevate(i);
            for (int i = 0; i < 1; i++)
            {
                domain1->KnotInsertion(0, 1.0 / 3);
                domain1->KnotInsertion(0, 2.0 / 3);
                domain1->KnotInsertion(1, 1.0 / 3);
                domain1->KnotInsertion(1, 2.0 / 3);
            }

            domain1->UniformRefine(j, 1);
            domain2->UniformRefine(j + 1, 1);
            domain3->UniformRefine(j, 1);
            array<shared_ptr<Surface<2, double>>, 3> cells;
            cells[0] = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, false, false, true});
            cells[0]->SurfaceInitialize();
            cells[1] = make_shared<Surface<2, double>>(domain2, array<bool, 4>{false, false, true, true});
            cells[1]->SurfaceInitialize();
            cells[2] = make_shared<Surface<2, double>>(domain3, array<bool, 4>{true, true, false, false});
            cells[2]->SurfaceInitialize();
            for (int i = 0; i < 2; i++)
            {
                for (int j = i + 1; j < 3; j++)
                    cells[i]->Match(cells[j]);
            }

            function<vector<double>(const VectorXd &)> body_force = [&b](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{8 * (1 + 3 * (6 - 10 * b + 5 * pow(b, 2)) * pow(x, 4) - 6 * y +
                                           3 * (8 - 6 * b + pow(b, 2)) * pow(y, 2) + 6 * (-3 + 2 * b) * pow(y, 3) + 3 * pow(y, 4) +
                                           pow(x, 3) * (-46 - 20 * pow(b, 2) - 60 * b * (-1 + y) + 60 * y) +
                                           3 * pow(x, 2) * (13 - 36 * y + 27 * pow(y, 2) - 6 * b * (2 - 4 * y + 5 * pow(y, 2)) + pow(b, 2) * (2 + 15 * pow(y, 2))) -
                                           6 * x * (2 - 9 * y + 16 * pow(y, 2) + 5 * pow(b, 2) * pow(y, 2) - 5 * pow(y, 3) + b * (-1 + 3 * y - 15 * pow(y, 2) + 5 * pow(y, 3))))};
            };

            function<vector<double>(const VectorXd &)> analytical_solution = [&b](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{pow(-1 + x, 2) * pow(x, 2) * pow(1 + (-1 + b) * x - y, 2) * pow(y, 2),
                                      2 * (-1 + x) * x * (1 + (-1 + b) * x - y) * pow(y, 2) *
                                          (-1 + 3 * (-1 + b) * pow(x, 2) + y - 2 * x * (-2 + b + y)),
                                      2 * pow(-1 + x, 2) * pow(x, 2) * (1 + (-1 + b) * x - 2 * y) * (1 + (-1 + b) * x - y) * y};
            };

            DofMapper<2, double> dof_map;
            BiharmonicMapper<2, double> mapper(dof_map);

            for (int i = 0; i < 3; i++)
            {
                cells[i]->Accept(mapper);
            }

            BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
            for (int i = 0; i < 3; i++)
            {
                cells[i]->Accept(stiffness);
            }
            SparseMatrix<double> stiffness_matrix_triangle_view, load_vector;
            stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
            stiffness.LoadAssembler(load_vector);
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();
            // BiharmonicInterfaceH1<2, double> interface(dof_map);
            BiharmonicInterface<2, double> interface1(dof_map);
            for (int i = 0; i < 3; i++)
            {
                // cells[i]->EdgeAccept(interface);
                cells[i]->EdgeAccept(interface1);
            }
            MatrixXd constraint, constraint1;
            // interface.ConstraintMatrix(constraint);
            interface1.ConstraintMatrix(constraint1);

            MatrixXd global_to_free = MatrixXd::Identity(dof_map.Dof(), dof_map.Dof());
            auto dirichlet_indices = dof_map.GlobalDirichletIndices();
            sort(dirichlet_indices.begin(), dirichlet_indices.end());
            for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); it++)
            {
                Accessory::removeRow(global_to_free, *it);
            }
            SparseMatrix<double> free_stiffness = SparseMatrix<double>(global_to_free.sparseView()) * stiffness_matrix * SparseMatrix<double>(global_to_free.transpose().sparseView());
            VectorXd free_load = global_to_free * load_vector;
            MatrixXd free_constraint = constraint1 * global_to_free.transpose();
            FullPivLU<MatrixXd> lu(free_constraint);
            lu.setThreshold(1e-13);
            MatrixXd ker = lu.kernel();
            SparseMatrix<double> ker_sparse = ker.sparseView();

            SparseMatrix<double> stiffness_sol = ker_sparse.transpose() * free_stiffness * ker_sparse;
            VectorXd load_sol = ker_sparse.transpose() * free_load;
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute(stiffness_sol);
            VectorXd Solution = cg.solve(load_sol);

            VectorXd solution = global_to_free.transpose() * ker_sparse * Solution;

            PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
            for (int i = 0; i < 3; i++)
            {
                cells[i]->Accept(post_process);
            }
            cout << post_process.L2Norm() << endl;
        }
        cout << endl;
    }
    // vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3;
    // solutionDomain1.push_back(domain1->KnotVectorGetter(0));
    // solutionDomain1.push_back(domain1->KnotVectorGetter(1));
    // solutionDomain2.push_back(domain2->KnotVectorGetter(0));
    // solutionDomain2.push_back(domain2->KnotVectorGetter(1));
    // solutionDomain3.push_back(domain3->KnotVectorGetter(0));
    // solutionDomain3.push_back(domain3->KnotVectorGetter(1));
    // VectorXd controlDomain1 = solution.segment(dof_map.StartingIndex(domain1), domain1->GetDof());
    // VectorXd controlDomain2 = solution.segment(dof_map.StartingIndex(domain2), domain2->GetDof());
    // VectorXd controlDomain3 = solution.segment(dof_map.StartingIndex(domain3), domain3->GetDof());
    // vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions(5);
    // solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain1, controlDomain1);
    // solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain2, controlDomain2);
    // solutions[2] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain3, controlDomain3);

    // double x, y;
    // ofstream file1, file2, file3;
    // file1.open("domain1.txt");
    // file2.open("domain2.txt");
    // file3.open("domain3.txt");
    // for (int i = 0; i <= 100; i++)
    // {
    //     for (int j = 0; j <= 100; j++)
    //     {
    //         double xi = 1.0 * i / 100, eta = 1.0 * j / 100;
    //         Vector2d u(xi, eta);
    //         VectorXd position1 = domain1->AffineMap(u);
    //         VectorXd position2 = domain2->AffineMap(u);
    //         VectorXd position3 = domain3->AffineMap(u);
    //         auto result1 = (solutions[0]->AffineMap(u)(0) - analytical_solution(position1)[0]);
    //         auto result2 = (solutions[1]->AffineMap(u)(0) - analytical_solution(position2)[0]);
    //         auto result3 = (solutions[2]->AffineMap(u)(0) - analytical_solution(position3)[0]);
    //         file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
    //         file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
    //         file3 << position3(0) << " " << position3(1) << " " << result3 << endl;
    //     }
    // }
    return 0;
}