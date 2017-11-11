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
using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
using VectorXmpf = Matrix<mpf_float_100, Dynamic, 1>;
using MatrixXmpf = Matrix<mpf_float_100, Dynamic, Dynamic>;
int main()
{
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

    GeometryVector points1({point1, point3, point2, point4});
    GeometryVector points2({point2, point4, point8, point6});
    GeometryVector points3({point4, point3, point6, point5});
    GeometryVector points4({point6, point5, point8, point7});
    GeometryVector points5({point3, point1, point5, point7});

    int degree, refine;
    cin >> degree >> refine;

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points2);
    auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points3);
    auto domain4 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points4);
    auto domain5 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points5);

    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain3->DegreeElevate(degree);
    domain4->DegreeElevate(degree);
    domain5->DegreeElevate(degree);
    domain3->KnotInsertion(1, 1.0 / 3);
    domain3->KnotInsertion(1, 2.0 / 3);
    domain3->KnotInsertion(0, 1.0 / 3);
    domain3->KnotInsertion(0, 2.0 / 3);

    domain1->UniformRefine(refine + 1);
    domain2->UniformRefine(refine + 1);
    domain3->UniformRefine(refine);
    domain4->UniformRefine(refine + 1);
    domain5->UniformRefine(refine + 1);
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

    const double pi = 3.14159265358979323846264338327;
    function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
        return vector<double>{4 * pow(pi, 4) * sin(pi * u(0)) * sin(pi * u(1))};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
        return vector<double>{sin(pi * u(0)) * sin(pi * u(1)), pi * cos(pi * u(0)) * sin(pi * u(1)), pi * sin(pi * u(0)) * cos(pi * u(1))};
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
    SparseMatrix<double> boundary_value, edge_constraint, vertex_constraint, constraint;
    boundary.CondensedDirichletBoundary(boundary_value);
    BiharmonicInterface<2, double> interface(dof_map);
    for (int i = 0; i < 5; i++)
    {
        cells[i]->EdgeAccept(interface);
    }
    interface.ConstraintMatrix(edge_constraint);
    BiharmonicVertexVisitor<2, double> vertex(dof_map);
    for (int i = 0; i < 5; i++)
    {
        cells[i]->VertexAccept(vertex);
    }
    vertex.ConstraintMatrix(vertex_constraint);
    constraint = (edge_constraint * vertex_constraint).pruned(1e-10);
    SparseMatrix<double> condensed_stiffness_matrix = global_to_condensed * constraint.transpose() * stiffness_matrix * constraint * global_to_condensed.transpose();
    SparseMatrix<double> free_stiffness_matrix = condensed_to_free * condensed_stiffness_matrix * condensed_to_free.transpose();
    SparseMatrix<double> condensed_rhs = global_to_condensed * constraint.transpose() * load_vector - condensed_stiffness_matrix * boundary_value;
    SparseMatrix<double> free_rhs = condensed_to_free * condensed_rhs;

    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(free_stiffness_matrix);
    VectorXd Solution = cg.solve(free_rhs);

    VectorXd solution = constraint * global_to_condensed.transpose() * (condensed_to_free.transpose() * Solution + boundary_value);
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
    VectorXd controlDomain1 = solution.segment(dof_map.StartingIndex(domain1), domain1->GetDof());
    VectorXd controlDomain2 = solution.segment(dof_map.StartingIndex(domain2), domain2->GetDof());
    VectorXd controlDomain3 = solution.segment(dof_map.StartingIndex(domain3), domain3->GetDof());
    VectorXd controlDomain4 = solution.segment(dof_map.StartingIndex(domain4), domain4->GetDof());
    VectorXd controlDomain5 = solution.segment(dof_map.StartingIndex(domain5), domain5->GetDof());
    vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions(5);
    solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain1, controlDomain1);
    solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain2, controlDomain2);
    solutions[2] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain3, controlDomain3);
    solutions[3] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain4, controlDomain4);
    solutions[4] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain5, controlDomain5);

    
    double x, y;
    ofstream file1, file2, file3, file4, file5;
    file1.open("domain1.txt");
    file2.open("domain2.txt");
    file3.open("domain3.txt");
    file4.open("domain4.txt");
    file5.open("domain5.txt");
    for (int i = 0; i <= 200; i++) {
        for (int j = 0; j <= 200; j++) {
            double xi = 1.0 * i / 200, eta = 1.0 * j / 200;
            Vector2d u(xi, eta);
            VectorXd position1 = domain1->AffineMap(u);
            VectorXd position2 = domain2->AffineMap(u);
            VectorXd position3 = domain3->AffineMap(u);
            VectorXd position4 = domain4->AffineMap(u);
            VectorXd position5 = domain5->AffineMap(u);
            auto result1 = abs(solutions[0]->AffineMap(u)(0) - analytical_solution(position1)[0]);
            auto result2 = abs(solutions[1]->AffineMap(u)(0) - analytical_solution(position2)[0]);
            auto result3 = abs(solutions[2]->AffineMap(u)(0) - analytical_solution(position3)[0]);
            auto result4 = abs(solutions[3]->AffineMap(u)(0) - analytical_solution(position4)[0]);
            auto result5 = abs(solutions[4]->AffineMap(u)(0) - analytical_solution(position5)[0]);
            file1 << position1(0) << " " << position1(1) << " " << result1 << endl;
            file2 << position2(0) << " " << position2(1) << " " << result2 << endl;
            file3 << position3(0) << " " << position3(1) << " " << result3 << endl;
            file4 << position4(0) << " " << position4(1) << " " << result4 << endl;
            file5 << position5(0) << " " << position5(1) << " " << result5 << endl;
        }
    }


    return 0;
}