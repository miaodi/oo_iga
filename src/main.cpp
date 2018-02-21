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
    double nu = .3;
    double E = 4.32e8;

    KnotVector<double> knot_vector;
    knot_vector.InitClosed(2, 0, 1);

    Vector3d point1(0, 0, 0), point2(0, .5, 0), point3(0, 1, 0), point4(.4, 0, 0), point5(.5, .5, 0), point6(.6, 1, 0), point7(1, 0, 0), point8(1, .5, 0), point9(1, 1, 0);
    Vector3d point10(0, 1, 0), point11(0, 1, -.5), point12(0, 1, -1), point13(.5, 1, 0), point14(.5, 1, -.5), point15(.5, 1, -1), point16(1, 1, 0), point17(1, 1, -.5), point18(1, 1, -1);
    GeometryVector points1{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    GeometryVector points2{point10, point11, point12, point13, point14, point15, point16, point17, point18};

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2);

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
        return vector<double>{0, 0, 0};
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

    auto south_indices = cells[0]->EdgePointerGetter(0)->Indices(1, 1);
    auto south_indices_slave = cells[1]->EdgePointerGetter(0)->Indices(1, 1);
    vector<int> dirichlet_indices;
    for (const auto &i : south_indices)
    {
        dirichlet_indices.push_back(master_start_index + 3 * i);
    }
    for (const auto &i : south_indices)
    {
        dirichlet_indices.push_back(master_start_index + 3 * i + 1);
    }
    for (const auto &i : south_indices)
    {
        dirichlet_indices.push_back(master_start_index + 3 * i + 2);
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
    load_vector.coeffRef(load_vector.rows() - 2, 0) = -1e-2;
    SparseMatrix<double> stiff_sol = sparse_global_to_free * constraint_matrix * stiffness_matrix * constraint_matrix.transpose() * sparse_global_to_free.transpose();
    SparseMatrix<double> load_sol = sparse_global_to_free * constraint_matrix * load_vector;
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(stiff_sol);
    VectorXd solution = constraint_matrix.transpose() * sparse_global_to_free.transpose() * cg.solve(load_sol);
    GeometryVector solution_ctrl_pts1, solution_ctrl_pts2;
    for (int i = 0; i < domain1->GetDof(); i++)
    {
        Vector3d temp;
        Vector3d ctrpt = domain1->CtrPtsGetter(i);
        temp << solution(3 * i + 0) + ctrpt(0), solution(3 * i + 1) + ctrpt(1), solution(3 * i + 2) + ctrpt(2);
        solution_ctrl_pts1.push_back(temp);
    }
    for (int i = 0; i < domain2->GetDof(); i++)
    {
        Vector3d temp;
        Vector3d ctrpt = domain2->CtrPtsGetter(i);
        temp << solution(slave_start_index + 3 * i + 0) + ctrpt(0), solution(slave_start_index + 3 * i + 1) + ctrpt(1), solution(slave_start_index + 3 * i + 2) + ctrpt(2);
        solution_ctrl_pts2.push_back(temp);
    }
    auto solution_domain1 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain1->KnotVectorGetter(0), domain1->KnotVectorGetter(1)}, solution_ctrl_pts1);
    auto solution_domain2 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain2->KnotVectorGetter(0), domain2->KnotVectorGetter(1)}, solution_ctrl_pts2);
    VectorXd u(2);
    u << 0, 1;
    cout << setprecision(10) << solution_domain2->AffineMap(u) << std::endl;
    u << 1, 1;
    cout << setprecision(10) << solution_domain2->AffineMap(u) << std::endl;
    // ofstream myfile1, myfile2;
    // myfile1.open("domain1.txt");
    // myfile2.open("domain2.txt");
    // for (int i = 0; i < 201; i++)
    // {
    //     for (int j = 0; j < 201; j++)
    //     {
    //         u << 1.0 * i / 200, 1.0 * j / 200;
    //         VectorXd result = solution_domain1->AffineMap(u);
    //         VectorXd position = domain1->AffineMap(u);
    //         myfile1 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << -result(1) << endl;
    //         result = solution_domain2->AffineMap(u);
    //         position = domain2->AffineMap(u);
    //         myfile2 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << -result(1) << endl;
    //     }
    // }

    // myfile1.close();
    // myfile2.close();
    for (int i = 0; i <= 20; i++)
    {
        u << 1.0 * i / 20, 1;
        VectorXd v(2);
        if (!Accessory::MapParametricPoint(&*domain1, u, &*domain2, v))
        {
            std::cout << "MapParametericPoint failed" << std::endl;
        }
        MatrixXd solution1_jacobian, solution2_jacobian;
        solution1_jacobian = solution_domain1->JacobianMatrix(u);
        solution2_jacobian = solution_domain2->JacobianMatrix(v);
        Vector3d xi_1, eta_1, xi_2, eta_2;
        xi_1 = solution1_jacobian.col(0);
        eta_1 = solution1_jacobian.col(1);
        xi_2 = solution2_jacobian.col(0);
        eta_2 = solution2_jacobian.col(1);
        Vector3d n1 = xi_1.cross(eta_1);
        Vector3d n2 = xi_2.cross(eta_2);
        n1.normalize();
        n2.normalize();
        cout << setprecision(10) << abs(acos(n1.dot(n2)) * 180.0 / boost::math::constants::pi<double>() - 90) / 90 << endl;
    }
    return 0;
}