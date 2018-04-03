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
#include "ConstraintAssembler.hpp"
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

    Vector3d point11(10.0, 0.0, 0.0), point12(10.0, 0, 10.0), point13(0.0, 0.0, 10.0), point14(10.0, 10.0, 0.0), point15(10.0, 10.0, 10.0), point16(0.0, 0.0, 10.0), point17(0.0, 10.0, 0.0), point18(0.0, 10.0, 10.0), point19(0.0, 0.0, 10.0);
    Vector3d point21(0.0, 10.0, 0.0), point22(0.0, 10.0, 10.0), point23(0.0, 0.0, 10.0), point24(-10.0, 10.0, 0.0), point25(-10.0, 10.0, 10.0), point26(0.0, 0.0, 10.0), point27(-10.0, 0.0, 0.0), point28(-10.0, 0, 10.0), point29(0.0, 0.0, 10.0);
    Vector3d point31(-10.0, 0.0, 0.0), point32(-10.0, 0, 10.0), point33(0.0, 0.0, 10.0), point34(-10.0, -10.0, 0.0), point35(-10.0, -10.0, 10.0), point36(0.0, 0.0, 10.0), point37(0.0, -10.0, 0.0), point38(0.0, -10.0, 10.0), point39(0.0, 0.0, 10.0);
    Vector3d point41(0.0, -10.0, 0.0), point42(0.0, -10.0, 10.0), point43(0.0, 0.0, 10.0), point44(10.0, -10.0, 0.0), point45(10.0, -10.0, 10.0), point46(0.0, 0.0, 10.0), point47(10.0, 0.0, 0.0), point48(10.0, 0, 10.0), point49(0.0, 0.0, 10.0);
    GeometryVector points1{point11, point12, point13, point14, point15, point16, point17, point18, point19};
    GeometryVector points2{point21, point22, point23, point24, point25, point26, point27, point28, point29};
    GeometryVector points3{point31, point32, point33, point34, point35, point36, point37, point38, point39};
    GeometryVector points4{point41, point42, point43, point44, point45, point46, point47, point48, point49};
    Vector1d weight1(1), weight2(1.0 / sqrt(2));
    WeightVector weights1{weight1, weight2, weight1, weight2, weight1 * 1.0 / 2, weight2, weight1, weight2, weight1};
    WeightVector weights2{weight1, weight2, weight1, weight2, weight1 * 1.0 / 2, weight2, weight1, weight2, weight1};
    WeightVector weights3{weight1, weight2, weight1, weight2, weight1 * 1.0 / 2, weight2, weight1, weight2, weight1};
    WeightVector weights4{weight1, weight2, weight1, weight2, weight1 * 1.0 / 2, weight2, weight1, weight2, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1, weights1);
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2, weights2);
    auto domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3, weights3);
    auto domain4 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4, weights4);

    int degree, refine;
    cin >> degree >> refine;
    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain3->DegreeElevate(degree);
    domain4->DegreeElevate(degree);

    domain1->KnotInsertion(0, .5);
    domain2->KnotInsertion(0, 1.0 / 3);
    domain2->KnotInsertion(0, 2.0 / 3);
    domain1->KnotInsertion(1, .5);
    domain2->KnotInsertion(1, 1.0 / 3);
    domain2->KnotInsertion(1, 2.0 / 3);
    domain4->KnotInsertion(0, .5);
    domain3->KnotInsertion(0, 1.0 / 3);
    domain3->KnotInsertion(0, 2.0 / 3);
    domain4->KnotInsertion(1, .5);
    domain3->KnotInsertion(1, 1.0 / 3);
    domain3->KnotInsertion(1, 2.0 / 3);

    domain1->UniformRefine(refine);
    domain2->UniformRefine(refine);
    domain3->UniformRefine(refine);
    domain4->UniformRefine(refine);
    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back(make_shared<Surface<3, double>>(domain1));
    cells[0]->SurfaceInitialize();
    cells.push_back(make_shared<Surface<3, double>>(domain2));
    cells[1]->SurfaceInitialize();
    cells.push_back(make_shared<Surface<3, double>>(domain3));
    cells[2]->SurfaceInitialize();
    cells.push_back(make_shared<Surface<3, double>>(domain4));
    cells[3]->SurfaceInitialize();
    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0, 0};
    };
    DofMapper dof;
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), 3 * i->GetDomain()->GetDof());
    }
    cout << dof.TotalDof() << endl;
    cells[0]->EdgePointerGetter(1)->Match(cells[1]->EdgePointerGetter(3));
    cells[1]->EdgePointerGetter(1)->Match(cells[2]->EdgePointerGetter(3));
    cells[2]->EdgePointerGetter(1)->Match(cells[3]->EdgePointerGetter(3));
    cells[3]->EdgePointerGetter(1)->Match(cells[0]->EdgePointerGetter(3));

    vector<Triplet<double>> constraint;
    ConstraintAssembler<double> constraint_assemble(dof);
    auto num_of_constraints = constraint_assemble.Assemble(cells, constraint);
    auto indices = cells[0]->EdgePointerGetter(2)->Indices(1, 1);
    auto start_index = dof.StartingDof(cells[0]->GetID());
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
        num_of_constraints++;
    }
    indices = cells[1]->EdgePointerGetter(2)->Indices(1, 1);
    start_index = dof.StartingDof(cells[1]->GetID());
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
        num_of_constraints++;
    }
    indices = cells[2]->EdgePointerGetter(2)->Indices(1, 1);
    start_index = dof.StartingDof(cells[2]->GetID());
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
        num_of_constraints++;
    }
    indices = cells[3]->EdgePointerGetter(2)->Indices(1, 1);
    start_index = dof.StartingDof(cells[3]->GetID());
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
        num_of_constraints++;
    }
    SparseMatrix<double> constraint_matrix(num_of_constraints, dof.TotalDof());
    constraint_matrix.setFromTriplets(constraint.begin(), constraint.end());
    constraint_matrix.pruned(1e-10);
    constraint_matrix.makeCompressed();
    MatrixXd dense_constraint = constraint_matrix;
    FullPivLU<MatrixXd> lu_decomp(dense_constraint);
    MatrixXd x = lu_decomp.kernel();

    // ColPivHouseholderQR<MatrixXd> qr(dense_constraint);
    // // Extract Q matrix
    // MatrixXd left = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Upper>();
    // MatrixXd right = qr.matrixR().topRightCorner(qr.rank(), constraint_matrix.cols() - qr.rank());
    // FullPivLU<MatrixXd> decomp(left);
    // SparseMatrix<double, RowMajor> x(dof.TotalDof(), right.cols());

    // MatrixXd right_sol = left.colPivHouseholderQr().solve(-right);
    // x.topRows(qr.rank()) = right_sol.sparseView();
    // SparseMatrix<double, RowMajor> identity(right.cols(), right.cols());
    // identity.setIdentity();
    // x.bottomRows(right.cols()) = identity;
    // x = qr.colsPermutation() * x;
    // x.pruned(1e-10);
    StiffnessAssembler<double> stiffness_assemble(dof);
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
    load_vector.coeffRef(dof.StartingDof(cells[0]->GetID()) + 3 * *(cells[0]->EdgePointerGetter(0)->Indices(1, 0).begin()), 0) = 2.0;
    load_vector.coeffRef(dof.StartingDof(cells[0]->GetID()) + 3 * *(cells[0]->EdgePointerGetter(0)->Indices(1, 0).end() - 1) + 1, 0) = -2.0;
    load_vector.coeffRef(dof.StartingDof(cells[2]->GetID()) + 3 * *(cells[2]->EdgePointerGetter(0)->Indices(1, 0).begin()), 0) = -2.0;
    load_vector.coeffRef(dof.StartingDof(cells[2]->GetID()) + 3 * *(cells[2]->EdgePointerGetter(0)->Indices(1, 0).end() - 1) + 1, 0) = 2.0;

    SparseMatrix<double> stiff_sol = (x.transpose() * stiffness_matrix * x).sparseView();
    stiff_sol.pruned(1e-10);
    SparseMatrix<double> load_sol = (x.transpose() * load_vector).sparseView();
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(stiff_sol);
    VectorXd solution = x * cg.solve(load_sol);

    GeometryVector solution_ctrl_pts1, solution_ctrl_pts2, solution_ctrl_pts3, solution_ctrl_pts4;
    for (int i = 0; i < domain1->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(3 * i + 0), solution(3 * i + 1), solution(3 * i + 2);
        solution_ctrl_pts1.push_back(temp);
    }
    for (int i = 0; i < domain2->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(dof.StartingDof(cells[1]->GetID()) + 3 * i + 0), solution(dof.StartingDof(cells[1]->GetID()) + 3 * i + 1), solution(dof.StartingDof(cells[1]->GetID()) + 3 * i + 2);
        solution_ctrl_pts2.push_back(temp);
    }
    for (int i = 0; i < domain3->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(dof.StartingDof(cells[2]->GetID()) + 3 * i + 0), solution(dof.StartingDof(cells[2]->GetID()) + 3 * i + 1), solution(dof.StartingDof(cells[2]->GetID()) + 3 * i + 2);
        solution_ctrl_pts3.push_back(temp);
    }
    for (int i = 0; i < domain4->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(dof.StartingDof(cells[3]->GetID()) + 3 * i + 0), solution(dof.StartingDof(cells[3]->GetID()) + 3 * i + 1), solution(dof.StartingDof(cells[3]->GetID()) + 3 * i + 2);
        solution_ctrl_pts4.push_back(temp);
    }
    auto solution_domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain1->KnotVectorGetter(0), domain1->KnotVectorGetter(1)}, solution_ctrl_pts1, domain1->WeightVectorGetter());
    auto solution_domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain2->KnotVectorGetter(0), domain2->KnotVectorGetter(1)}, solution_ctrl_pts2, domain2->WeightVectorGetter());
    auto solution_domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain3->KnotVectorGetter(0), domain3->KnotVectorGetter(1)}, solution_ctrl_pts3, domain3->WeightVectorGetter());
    auto solution_domain4 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain4->KnotVectorGetter(0), domain4->KnotVectorGetter(1)}, solution_ctrl_pts4, domain4->WeightVectorGetter());
    Vector2d u;
    u << 0, 0;
    cout << setprecision(10) << solution_domain1->AffineMap(u) << std::endl;
    u << 1, 0;
    cout << setprecision(10) << solution_domain1->AffineMap(u) << std::endl;

    // ofstream myfile1, myfile2, myfile3, myfile4;
    // myfile1.open("domain1.txt");
    // myfile2.open("domain2.txt");
    // myfile3.open("domain3.txt");
    // myfile4.open("domain4.txt");

    // for (int i = 0; i < 51; i++)
    // {
    //     for (int j = 0; j < 51; j++)
    //     {
    //         u << 1.0 * i / 50, 1.0 * j / 50;
    //         VectorXd result = solution_domain1->AffineMap(u);
    //         VectorXd position = domain1->AffineMap(u);
    //         myfile1 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << result(0) << endl;
    //         result = solution_domain2->AffineMap(u);
    //         position = domain2->AffineMap(u);
    //         myfile2 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << result(0) << endl;
    //         result = solution_domain3->AffineMap(u);
    //         position = domain3->AffineMap(u);
    //         myfile3 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << result(0) << endl;
    //         result = solution_domain4->AffineMap(u);
    //         position = domain4->AffineMap(u);
    //         myfile4 << 20 * result(0) + position(0) << " " << 20 * result(1) + position(1) << " " << 20 * result(2) + position(2) << " " << result(0) << endl;
    //     }
    // }
    // myfile1.close();
    // myfile2.close();
    // myfile3.close();
    // myfile4.close();
    return 0;
}