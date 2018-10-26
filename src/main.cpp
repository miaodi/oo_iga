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
#include "ProjectionStiffnessVisitor.hpp"
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
using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
using VectorXmpf = Matrix<mpf_float_100, Dynamic, 1>;
using MatrixXmpf = Matrix<mpf_float_100, Dynamic, Dynamic>;
int main()
{
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 1);
    Vector2d point2(0, 0);
    Vector2d point3(.16, .81);
    Vector2d point4(.18, .18);
    Vector2d point5(.77, .82);
    Vector2d point6(.81, .17);
    Vector2d point7(1, 1);
    Vector2d point8(1, 0);

    GeometryVector points1({point1, point3, point2, point4});
    GeometryVector points2({point2, point4, point8, point6});
    GeometryVector points3({point4, point3, point6, point5});
    GeometryVector points4({point6, point5, point8, point7});
    GeometryVector points5({point3, point1, point5, point7});

    int degree, refine;
    cin >> degree >> refine;
    for (int i = 1; i < degree; i++)
    {
        for (int j = 1; j < refine; j++)
        {

            auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
            auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points2);
            auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points3);
            auto domain4 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points4);
            auto domain5 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points5);

            domain1->DegreeElevate(i);
            domain2->DegreeElevate(i);
            domain3->DegreeElevate(i);
            domain4->DegreeElevate(i);
            domain5->DegreeElevate(i);
            for (int k = 0; k < 1; k++)
            {
                domain2->KnotInsertion(0, 1.0 / 3);
                domain2->KnotInsertion(0, 2.0 / 3);
                domain2->KnotInsertion(1, 1.0 / 3);
                domain2->KnotInsertion(1, 2.0 / 3);
            }
            for (int k = 0; k < 1; k++)
            {
                domain3->KnotInsertion(0, 1.0 / 5);
                domain3->KnotInsertion(0, 2.0 / 5);
                domain3->KnotInsertion(0, 3.0 / 5);
                domain3->KnotInsertion(0, 4.0 / 5);
                domain3->KnotInsertion(1, 1.0 / 5);
                domain3->KnotInsertion(1, 2.0 / 5);
                domain3->KnotInsertion(1, 3.0 / 5);
                domain3->KnotInsertion(1, 4.0 / 5);
            }
            for (int k = 0; k < 1; k++)
            {
                domain5->KnotInsertion(0, 1.0 / 3);
                domain5->KnotInsertion(0, 2.0 / 3);
                domain5->KnotInsertion(1, 1.0 / 3);
                domain5->KnotInsertion(1, 2.0 / 3);
            }
            domain1->UniformRefine(j + 1, 1);
            domain2->UniformRefine(j, 1);
            domain3->UniformRefine(j, 1);
            domain4->UniformRefine(j + 1, 1);
            domain5->UniformRefine(j, 1);
            array<shared_ptr<Surface<2, double>>, 5> cells;
            cells[0] = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, false, false, false});
            cells[0]->SurfaceInitialize();
            cells[1] = make_shared<Surface<2, double>>(domain2, array<bool, 4>{true, false, false, false});
            cells[1]->SurfaceInitialize();
            cells[2] = make_shared<Surface<2, double>>(domain3, array<bool, 4>{false, false, false, false});
            cells[2]->SurfaceInitialize();
            cells[3] = make_shared<Surface<2, double>>(domain4, array<bool, 4>{false, true, false, false});
            cells[3]->SurfaceInitialize();
            cells[4] = make_shared<Surface<2, double>>(domain5, array<bool, 4>{false, false, true, false});
            cells[4]->SurfaceInitialize();

            for (int i = 0; i < 4; i++)
            {
                for (int j = i + 1; j < 5; j++)
                    cells[i]->Match(cells[j]);
            }

            const double Pi = 3.14159265358979323846264338327;
            function<vector<double>(const VectorXd &)> body_force = [&](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{-64 * pow(Pi, 4) *
                                      (cos(4 * Pi * x) - 2 * cos(4 * Pi * (x - y)) + cos(4 * Pi * y) -
                                       2 * cos(4 * Pi * (x + y)))};
            };

            function<vector<double>(const VectorXd &)> analytical_solution = [&](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{
                    pow(sin(2 * Pi * x), 2) * pow(sin(2 * Pi * y), 2),
                    4 * Pi * cos(2 * Pi * x) * sin(2 * Pi * x) * pow(sin(2 * Pi * y), 2),
                    4 * Pi * cos(2 * Pi * y) * pow(sin(2 * Pi * x), 2) * sin(2 * Pi * y),
                    8 * pow(Pi, 2) * pow(cos(2 * Pi * x), 2) * pow(sin(2 * Pi * y), 2) -
                        8 * pow(Pi, 2) * pow(sin(2 * Pi * x), 2) * pow(sin(2 * Pi * y), 2),
                    16 * pow(Pi, 2) * cos(2 * Pi * x) * cos(2 * Pi * y) * sin(2 * Pi * x) * sin(2 * Pi * y),
                    8 * pow(Pi, 2) * pow(cos(2 * Pi * y), 2) * pow(sin(2 * Pi * x), 2) -
                        8 * pow(Pi, 2) * pow(sin(2 * Pi * x), 2) * pow(sin(2 * Pi * y), 2)};
            };

            DofMapper<2, double> dof_map;
            BiharmonicMapper<2, double> mapper(dof_map);

            for (int i = 0; i < 5; i++)
            {
                cells[i]->Accept(mapper);
            }

            SparseMatrix<double> global_to_condensed, condensed_to_free, global_to_free;
            dof_map.CondensedIndexMap(global_to_condensed);
            dof_map.FreeIndexMap(global_to_free);
            dof_map.FreeToCondensedIndexMap(condensed_to_free);

            BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
            for (int i = 0; i < 5; i++)
            {
                cells[i]->Accept(stiffness);
            }
            SparseMatrix<double> stiffness_matrix_triangle_view, load_vector;
            stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
            stiffness.LoadAssembler(load_vector);
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();

            BiharmonicDirichletBoundaryVisitor<2, double> boundary(dof_map, analytical_solution);
            for (int i = 0; i < 5; i++)
            {
                cells[i]->EdgeAccept(boundary);
            }
            SparseMatrix<double> boundary_value, edge_constraint, edge_constraint1, vertex_constraint, constraint;
            boundary.CondensedDirichletBoundary(boundary_value);
            // BiharmonicInterfaceH1<2, double> interface(dof_map);
            BiharmonicInterface<2, double> interface1(dof_map);
            for (int i = 0; i < 5; i++)
            {
                // cells[i]->EdgeAccept(interface);
                cells[i]->EdgeAccept(interface1);
            }
            // interface.ConstraintMatrix(edge_constraint);
            interface1.ConstraintMatrix(edge_constraint1);
            BiharmonicVertexVisitor<2, double> vertex(dof_map);
            for (int i = 0; i < 5; i++)
            {
                cells[i]->VertexAccept(vertex);
            }

            vertex.ConstraintMatrix(vertex_constraint);
            constraint = (edge_constraint1 * vertex_constraint).pruned(1e-11);
            SparseMatrix<double> condensed_stiffness_matrix = global_to_condensed * constraint.transpose() * stiffness_matrix * constraint * global_to_condensed.transpose();
            SparseMatrix<double> free_stiffness_matrix = condensed_to_free * condensed_stiffness_matrix * condensed_to_free.transpose();
            SparseMatrix<double> condensed_rhs = global_to_condensed * constraint.transpose() * load_vector - condensed_stiffness_matrix * boundary_value;
            SparseMatrix<double> free_rhs = condensed_to_free * condensed_rhs;

            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute(free_stiffness_matrix);
            VectorXd Solution = cg.solve(free_rhs);

            VectorXd solution = constraint * global_to_condensed.transpose() * (condensed_to_free.transpose() * Solution + boundary_value);

            PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
            for (int i = 0; i < 5; i++)
            {
                cells[i]->Accept(post_process);
            }
            cout << "L2: " << post_process.L2Norm() << "H2: " << post_process.H2Norm() << endl;
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