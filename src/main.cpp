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
#include <spectra/GenEigsSolver.h>
#include <spectra/MatOp/SparseGenMatProd.h>

using namespace Eigen;
using namespace Spectra;
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
    Vector2d point1(-sqrt(3), -1), point2(-sqrt(3) / 2, .5), point3(0, 2), point4(0, -1), point5(0, 0), point6(sqrt(3) / 2, .5), point7(sqrt(3), -1);

    GeometryVector point{point1, point2, point4, point5};
    GeometryVector pointt{point2, point3, point5, point6};
    GeometryVector pointtt{point4, point5, point7, point6};

    int degree, refine;
    cin >> degree >> refine;

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, point);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointt);
    auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointtt);

    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain3->DegreeElevate(degree);

    domain1->UniformRefine(refine, 1);
    domain2->UniformRefine(refine, 1);
    domain3->UniformRefine(refine, 1);
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

    const double pi = 3.14159265358979323846264338327;
    function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
        return vector<double>{144 * pow(-3 * u(0) + sqrt(3) * (u(1) - 2), 2) + 144 * pow(3 * u(0) + sqrt(3) * (u(1) - 2), 2) + 1728 * pow(u(1) + 1, 2)};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{pow((u(1) + 1) * (sqrt(3) * (u(1) - 2) - 3 * u(0)) * (sqrt(3) * (u(1) - 2) + 3 * u(0)), 2),
                              36 * x * (3 * x - sqrt(3) * (y - 2)) * (3 * x + sqrt(3) * (y - 2)) * pow(1 + y, 2),
                              18 * (3 * x - sqrt(3) * (y - 2)) * (3 * x + sqrt(3) * (y - 2)) * (1 + y) * (x * x - (y - 2) * y)};
    };

    DofMapper<2, double> dof_map;
    BiharmonicMapper<2, double> mapper(dof_map);

    for (int i = 0; i < 3; i++)
    {
        cells[i]->Accept(mapper);
    }

    SparseMatrix<double> global_to_condensed, condensed_to_free, global_to_free;
    dof_map.CondensedIndexMap(global_to_condensed);
    dof_map.FreeIndexMap(global_to_free);
    dof_map.FreeToCondensedIndexMap(condensed_to_free);

    BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
    for (int i = 0; i < 3; i++)
    {
        cells[i]->Accept(stiffness);
    }
    SparseMatrix<double> stiffness_matrix_triangle_view, sparse_load_vector;
    stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
    stiffness.LoadAssembler(sparse_load_vector);
    SparseMatrix<double> sparse_stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();
    MatrixXd stiffness_matrix = MatrixXd(sparse_stiffness_matrix);
    VectorXd load_vector = VectorXd(sparse_load_vector);
    BiharmonicDirichletBoundaryVisitor<2, double> boundary(dof_map, analytical_solution);
    for (int i = 0; i < 3; i++)
    {
        cells[i]->EdgeAccept(boundary);
    }
    MatrixXd boundary_matrix;
    VectorXd boundary_vector;
    boundary.DirichletBoundary(boundary_matrix, boundary_vector);
    BiharmonicInterface<2, double> interface(dof_map);
    for (int i = 0; i < 3; i++)
    {
        cells[i]->EdgeAccept(interface);
    }
    MatrixXd edge_constraint;
    interface.ConstraintMatrix(edge_constraint);

    stiffness_matrix.conservativeResize(dof_map.Dof() + boundary_matrix.rows(), dof_map.Dof() + boundary_matrix.rows());
    load_vector.conservativeResize(dof_map.Dof() + boundary_matrix.rows() + edge_constraint.rows());

    stiffness_matrix.block(dof_map.Dof(), 0, boundary_matrix.rows(), dof_map.Dof()) = boundary_matrix;
    // load_vector.segment(dof_map.Dof(), boundary_matrix.rows()) = boundary_vector;
    // stiffness_matrix.block(dof_map.Dof() + boundary_matrix.rows(), 0, edge_constraint.rows(), dof_map.Dof()) = edge_constraint;
    // load_vector.segment(dof_map.Dof() + boundary_matrix.rows(), edge_constraint.rows()) = VectorXd::Zero(edge_constraint.rows());
    stiffness_matrix = stiffness_matrix.selfadjointView<Eigen::Lower>();
    EigenSolver<MatrixXd> es(stiffness_matrix);
    cout << es.eigenvalues() << endl;
    // BiharmonicVertexVisitor<2, double> vertex(dof_map);
    // for (int i = 0; i < 3; i++)
    // {
    //     cells[i]->VertexAccept(vertex);
    // }
    // vertex.ConstraintMatrix(vertex_constraint);
    // constraint = (edge_constraint * vertex_constraint).pruned(1e-10);
    // SparseMatrix<double> condensed_stiffness_matrix = global_to_condensed * constraint.transpose() * stiffness_matrix * constraint * global_to_condensed.transpose();
    // SparseMatrix<double> free_stiffness_matrix = condensed_to_free * condensed_stiffness_matrix * condensed_to_free.transpose();
    // SparseMatrix<double> condensed_rhs = global_to_condensed * constraint.transpose() * load_vector - condensed_stiffness_matrix * boundary_value;
    // SparseMatrix<double> free_rhs = condensed_to_free * condensed_rhs;

    // ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    // cg.compute(free_stiffness_matrix);
    // VectorXd Solution = cg.solve(free_rhs);

    // VectorXd solution = Solution.segment(0, dof_map.Dof());

    // PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
    // for (int i = 0; i < 3; i++)
    // {
    //     cells[i]->Accept(post_process);
    // }
    // cout << post_process.L2Norm() << endl;
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
    // for (int i = 0; i <= 300; i++)
    // {
    //     for (int j = 0; j <= 300; j++)
    //     {
    //         double xi = 1.0 * i / 300, eta = 1.0 * j / 300;
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

    // return 0;
}