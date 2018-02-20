#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "MembraneStiffnessVisitor.hpp"
#include "BendingStiffnessVisitor.hpp"
#include "PostProcess.hpp"
#include "H1DomainSemiNormVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include "DofMapper.hpp"
#include <fstream>
#include <time.h>
#include <boost/math/constants/constants.hpp>
#include "BiharmonicInterfaceVisitor.hpp"
#include "StiffnessAssembler.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 3, double>::GeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 3, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

int main()
{
    double nu = .0;
    double E = 4.32e8;
    double R = 25;
    double L = 50;
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(2, 0, 1);
    double rad = 40.0 / 180 * boost::math::constants::pi<double>();
    double a = sin(rad) * R;
    double b = a * tan(rad);
    Vector3d point1(-a, 0, 0), point2(-a, L / 4, 0), point3(-a, L / 2, 0), point4(0, 0, b), point5(0, L / 4, b), point6(0, L / 2, b), point7(a, 0, 0), point8(a, L / 4, 0), point9(a, L / 2, 0);
    Vector3d point10(-a, L / 2, 0), point11(-a, 3.0 * L / 4, 0), point12(-a, L, 0), point13(0, L / 2, b), point14(0, 3.0 * L / 4, b), point15(0, L, b), point16(a, L / 2, 0), point17(a, 3.0 * L / 4, 0), point18(a, L, 0);
    GeometryVector points1{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    GeometryVector points2{point10, point11, point12, point13, point14, point15, point16, point17, point18};

    Vector1d weight1(1), weight2(sin(boost::math::constants::pi<double>() / 2 - rad));
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1, weights);
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2, weights);

    int degree, refine;
    cin >> degree >> refine;
    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain1->KnotInsertion(0, .5);
    domain2->KnotInsertion(0, 1.0 / 3);
    domain2->KnotInsertion(0, 2.0 / 3);
    domain1->KnotInsertion(1, .5);
    domain2->KnotInsertion(1, 1.0 / 3);
    domain2->KnotInsertion(1, 2.0 / 3);
    domain1->UniformRefine(refine);
    domain2->UniformRefine(refine);
    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back(make_shared<Surface<3, double>>(domain1));
    cells[0]->SurfaceInitialize();
    cells.push_back(make_shared<Surface<3, double>>(domain2));
    cells[1]->SurfaceInitialize();
    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0, -90};
    };
    DofMapper dof;
    cells[0]->Match(cells[1]);
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), 3 * i->GetDomain()->GetDof());
    }
    BiharmonicInterfaceVisitor<3, double> biharmonic_interface;
    cells[0]->EdgeAccept(biharmonic_interface);
    cells[1]->EdgeAccept(biharmonic_interface);
    StiffnessAssembler<double> stiffness_assemble(dof);
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
    auto constraint = biharmonic_interface.GetConstraintData();
    // constraint.Print();
    SparseMatrix<double> constraint_matrix, identity;
    constraint_matrix.resize(dof.TotalDof(), dof.TotalDof());
    identity.resize(3, 3);
    constraint_matrix.setIdentity();
    identity.setIdentity();
    int master_start_index, slave_start_index;
    master_start_index = dof.StartingDof(biharmonic_interface.MasterID());
    slave_start_index = dof.StartingDof(biharmonic_interface.SlaveID());
    for (int i = 0; i < constraint._rowIndices->size(); ++i)
    {
        for (int j = 0; j < constraint._colIndices->size(); ++j)
        {
            constraint_matrix.coeffRef(master_start_index + (*constraint._colIndices)[j], slave_start_index + (*constraint._rowIndices)[i]) = (*constraint._matrix)(i, j);
        }
    }

    auto south_indices = cells[0]->EdgePointerGetter(0)->Indices(1, 0);
    auto south_indices_slave = cells[1]->EdgePointerGetter(0)->Indices(1, 1);
    auto north_indices = cells[1]->EdgePointerGetter(2)->Indices(1, 0);
    vector<int> dirichlet_indices;
    for (const auto &i : south_indices)
    {
        dirichlet_indices.push_back(master_start_index + 3 * i);
    }

    dirichlet_indices.push_back(master_start_index + 3 * *(south_indices.begin()) + 1);
    for (const auto &i : south_indices)
    {
        dirichlet_indices.push_back(master_start_index + 3 * i + 2);
    }
    for (const auto &i : north_indices)
    {
        dirichlet_indices.push_back(slave_start_index + 3 * i);
    }

    for (const auto &i : north_indices)
    {
        dirichlet_indices.push_back(slave_start_index + 3 * i + 2);
    }

    for (const auto &i : south_indices_slave)
    {
        dirichlet_indices.push_back(slave_start_index + 3 * i + 0);
    }
    for (const auto &i : south_indices_slave)
    {
        dirichlet_indices.push_back(slave_start_index + 3 * i + 1);
    }
    for (const auto &i : south_indices_slave)
    {
        dirichlet_indices.push_back(slave_start_index + 3 * i + 2);
    }

    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    MatrixXd global_to_free = MatrixXd::Identity(dof.TotalDof(), dof.TotalDof());
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    {
        Accessory::removeRow(global_to_free, *it);
    }

    SparseMatrix<double> sparse_global_to_free = global_to_free.sparseView();
    SparseMatrix<double> stiff_sol = sparse_global_to_free * constraint_matrix * stiffness_matrix * constraint_matrix.transpose() * sparse_global_to_free.transpose();
    SparseMatrix<double> load_sol = sparse_global_to_free * constraint_matrix * load_vector;
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(stiff_sol);
    VectorXd solution = constraint_matrix.transpose() * sparse_global_to_free.transpose() * cg.solve(load_sol);
    GeometryVector solution_ctrl_pts1, solution_ctrl_pts2;
    for (int i = 0; i < domain1->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(3 * i + 0), solution(3 * i + 1), solution(3 * i + 2);
        solution_ctrl_pts1.push_back(temp);
    }
    for (int i = 0; i < domain2->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(slave_start_index + 3 * i + 0), solution(slave_start_index + 3 * i + 1), solution(slave_start_index + 3 * i + 2);
        solution_ctrl_pts2.push_back(temp);
    }
    auto solution_domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain1->KnotVectorGetter(0), domain1->KnotVectorGetter(1)}, solution_ctrl_pts1, domain1->WeightVectorGetter());
    auto solution_domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain2->KnotVectorGetter(0), domain2->KnotVectorGetter(1)}, solution_ctrl_pts2, domain2->WeightVectorGetter());
    Vector2d u;
    u << 1, 1;
    cout << setprecision(10) << solution_domain1->AffineMap(u) << std::endl;
    u << 1, 0;
    cout << setprecision(10) << solution_domain2->AffineMap(u) << std::endl;
    ofstream file;
    file.open("stiff.txt");
    file << MatrixXd(sparse_global_to_free * constraint_matrix);
    return 0;
}