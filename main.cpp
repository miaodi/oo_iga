#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorNURBSBasis.h"
#include "Topology.hpp"
#include "Surface.hpp"
#include "Vertex.hpp"
#include "DofMapper.hpp"
#include "PoissonMapper.hpp"
#include "BiharmonicMapper.hpp"
#include "PoissonStiffnessVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "PoissonDirichletBoundaryVisitor.hpp"
#include "BiharmonicDirichletBoundaryVisitor.hpp"
#include "PoissonVertexVisitor.hpp"
#include "PoissonInterface.hpp"
#include "BiharmonicInterface.hpp"
#include "PostProcess.hpp"
#include <fstream>
#include <time.h>

using namespace Eigen;
using namespace std;
using namespace Accessory;
using KnotSpanList = TensorBsplineBasis<2, double>::KnotSpanList;

using Vector1d = Matrix<double, 1, 1>;

int main()
{
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0), point2(0, 2), point3(1, 1), point4(1, 2), point5(2, 0), point6(2, 1), point7(2, 2);

    vector<Vector2d> point{point1, point2, point3, point4};
    vector<Vector2d> pointt{point1, point3, point5, point6};
    vector<Vector2d> pointtt{point3, point4, point6, point7};

    int order, refine;
    cin >> order >> refine;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < order; i++)
    {
        for (int j = 0; j < refine; j++)
        {
            auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, point);
            auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, pointt);
            auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, pointtt);
            domain1->DegreeElevate(i);
            domain2->DegreeElevate(i);
            domain3->DegreeElevate(i);
            domain1->UniformRefine(1);
            domain2->UniformRefine(1);
            domain3->KnotInsertion(0, 1.0 / 3);
            domain3->KnotInsertion(0, 2.0 / 3);
            domain3->KnotInsertion(1, 1.0 / 3);
            domain3->KnotInsertion(1, 2.0 / 3);

            domain1->UniformRefine(j);
            domain2->UniformRefine(j);
            domain3->UniformRefine(j);
            auto surface1 = make_shared<Surface<2, double>>(domain1, array<bool, 4>{false, false, true, true});
            surface1->SurfaceInitialize();
            auto surface2 = make_shared<Surface<2, double>>(domain2, array<bool, 4>{true, true, false, false});
            surface2->SurfaceInitialize();
            auto surface3 = make_shared<Surface<2, double>>(domain3, array<bool, 4>{false, true, true, false});
            surface3->SurfaceInitialize();
            vector<shared_ptr<Surface<2, double>>> cells(3);
            cells[0] = surface1;
            cells[1] = surface2;
            cells[2] = surface3;
            surface1->Match(surface2);
            surface1->Match(surface3);
            surface2->Match(surface3);

            const double pi = 3.14159265358979323846264338327;
            function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
                return vector<double>{72 * sin(6 * u(0)) * sin(6 * u(1))};
            };

            function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
                return vector<double>{sin(6 * u(0)) * sin(6 * u(1)), cos(u(0)) * sin(u(1)), sin(u(0)) * cos(u(1))};
            };

            DofMapper<2, double> dof_map;
            PoissonMapper<2, double> mapper(dof_map);
            surface1->Accept(mapper);
            surface2->Accept(mapper);
            surface3->Accept(mapper);

            SparseMatrix<double> global_to_condensed, condensed_to_free, global_to_free;
            dof_map.CondensedIndexMap(global_to_condensed);
            dof_map.FreeIndexMap(global_to_free);
            dof_map.FreeToCondensedIndexMap(condensed_to_free);

            PoissonStiffnessVisitor<2, double> stiffness(dof_map, body_force);
            surface1->Accept(stiffness);
            surface2->Accept(stiffness);
            surface3->Accept(stiffness);
            SparseMatrix<double> stiffness_matrix_triangle_view, load_vector;
            stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
            stiffness.LoadAssembler(load_vector);
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();

            PoissonDirichletBoundaryVisitor<2, double> boundary(dof_map, analytical_solution);
            surface1->EdgeAccept(boundary);
            surface2->EdgeAccept(boundary);
            surface3->EdgeAccept(boundary);
            SparseMatrix<double> boundary_value, edge_constraint, vertex_constraint, constraint;
            boundary.CondensedDirichletBoundary(boundary_value);
            PoissonInterface<2, double> interface(dof_map);
            surface1->EdgeAccept(interface);
            surface2->EdgeAccept(interface);
            surface3->EdgeAccept(interface);
            PoissonVertexVisitor<2, double> vertex(dof_map);
            surface1->VertexAccept(vertex);
            surface2->VertexAccept(vertex);
            surface3->VertexAccept(vertex);
            interface.ConstraintMatrix(edge_constraint);
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
            PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
            surface1->Accept(post_process);
            surface2->Accept(post_process);
            surface3->Accept(post_process);
            cout<<post_process.L2Norm()<<endl;
        }
        std::cout << endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "test function took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
    return 0;
}