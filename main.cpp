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
#include "PoissonInterface.hpp"
#include "BiharmonicInterface.hpp"

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

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, point);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, pointt);
    auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(vector<KnotVector<double>>{a, a}, pointtt);
    domain1->DegreeElevate(3);
    domain2->DegreeElevate(3);
    domain3->DegreeElevate(3);

    domain1->UniformRefine(1);
    domain2->UniformRefine(1);
    domain3->KnotInsertion(0, 1.0 / 3);
    domain3->KnotInsertion(0, 2.0 / 3);
    domain3->KnotInsertion(1, 1.0 / 3);
    domain3->KnotInsertion(1, 2.0 / 3);

    domain1->UniformRefine(1);
    domain2->UniformRefine(1);
    domain3->UniformRefine(1);
    auto surface1 = make_shared<Surface<2, double>>(domain1, array<bool, 4>{false, false, true, true});
    surface1->SurfaceInitialize();
    auto surface2 = make_shared<Surface<2, double>>(domain2, array<bool, 4>{true, true, false, false});
    surface2->SurfaceInitialize();
    auto surface3 = make_shared<Surface<2, double>>(domain3, array<bool, 4>{false, true, true, false});
    surface3->SurfaceInitialize();

    surface1->Match(surface2);
    surface1->Match(surface3);
    surface2->Match(surface3);
    surface1->PrintVertexInfo();
    surface2->PrintVertexInfo();
    surface3->PrintVertexInfo();

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{4 * sin(u(0)) * sin(u(1))};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [](const VectorXd &u) {
        return vector<double>{sin(u(0)) * sin(u(1)), cos(u(0)) * sin(u(1)), sin(u(0)) * cos(u(1))};
    };

    DofMapper<2, double> dof_map;
    BiharmonicMapper<2, double> mapper(dof_map);
    surface1->Accept(mapper);
    surface2->Accept(mapper);
    surface3->Accept(mapper);
    SparseMatrix<double> global_to_condensed, condensed_to_free, global_to_free;
    dof_map.CondensedIndexMap(global_to_condensed);
    dof_map.FreeIndexMap(global_to_free);
    dof_map.FreeToCondensedIndexMap(condensed_to_free);

    BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
    surface1->Accept(stiffness);
    surface2->Accept(stiffness);
    surface3->Accept(stiffness);
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness.StiffnessAssembler(stiffness_matrix);
    stiffness.LoadAssembler(load_vector);

    BiharmonicDirichletBoundaryVisitor<2, double> boundary(dof_map, analytical_solution);
    surface1->EdgeAccept(boundary);
    surface2->EdgeAccept(boundary);
    surface3->EdgeAccept(boundary);
    SparseMatrix<double> boundary_value, constraint;
    boundary.CondensedDirichletBoundary(boundary_value);
    BiharmonicInterface<2, double> interface(dof_map);
    surface1->EdgeAccept(interface);
    surface2->EdgeAccept(interface);
    surface3->EdgeAccept(interface);
    interface.ConstraintMatrix(constraint);
    for (int k = 0; k < constraint.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(constraint, k); it; ++it)
        {
            cout<<it.value()<<", "<<it.row()<<", "<<it.col()<<endl;
        }
    SparseMatrix<double> condensed_stiffness_matrix = global_to_condensed * constraint.transpose() * stiffness_matrix * constraint * global_to_condensed.transpose();
    SparseMatrix<double> free_stiffness_matrix = condensed_to_free * condensed_stiffness_matrix * condensed_to_free.transpose();
    SparseMatrix<double> condensed_rhs = global_to_condensed * constraint.transpose() * load_vector - condensed_stiffness_matrix * boundary_value;
    SparseMatrix<double> free_rhs = condensed_to_free * condensed_rhs;
    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(free_stiffness_matrix);
    VectorXd Solution = cg.solve(free_rhs);
    return 0;
}