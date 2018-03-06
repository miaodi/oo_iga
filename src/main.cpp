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

    Vector3d point11(-16.06969024216348, 0.0, 0.0), point12(-16.06969024216348, 12.5, 0.0), point13(-16.06969024216348, 25.0, 0.0), point14(-9.09925585665506, 0.0, 5.84888892202555), point15(-9.09925585665506, 12.5, 5.84888892202555), point16(-9.09925585665506, 25.0, 5.84888892202555), point17(0.0, 0.0, 5.84888892202555), point18(0.0, 12.5, 5.84888892202555), point19(0.0, 25.0, 5.84888892202555);
    Vector3d point21(-16.06969024216348, 25.0, 0.0), point22(-16.06969024216348, 37.5, 0.0), point23(-16.06969024216348, 50.0, 0.0), point24(-9.09925585665506, 25.0, 5.84888892202555), point25(-9.09925585665506, 37.5, 5.84888892202555), point26(-9.09925585665506, 50.0, 5.84888892202555), point27(0.0, 25.0, 5.84888892202555), point28(0.0, 37.5, 5.84888892202555), point29(0.0, 50.0, 5.84888892202555);
    Vector3d point31(0.0, 0.0, 5.84888892202555), point32(0.0, 12.5, 5.84888892202555), point33(0.0, 25.0, 5.84888892202555), point34(9.09925585665506, 0.0, 5.84888892202555), point35(9.09925585665506, 12.5, 5.84888892202555), point36(9.09925585665506, 25.0, 5.84888892202555), point37(16.06969024216348, 0.0, 0.0), point38(16.06969024216348, 12.5, 0.0), point39(16.06969024216348, 25.0, 0.0);
    Vector3d point41(0.0, 25.0, 5.84888892202555), point42(0.0, 37.5, 5.84888892202555), point43(0.0, 50.0, 5.84888892202555), point44(9.09925585665506, 25.0, 5.84888892202555), point45(9.09925585665506, 37.5, 5.84888892202555), point46(9.09925585665506, 50.0, 5.84888892202555), point47(16.06969024216348, 25.0, 0.0), point48(16.06969024216348, 37.5, 0.0), point49(16.06969024216348, 50.0, 0.0);
    GeometryVector points1{point11, point12, point13, point14, point15, point16, point17, point18, point19};
    GeometryVector points2{point21, point22, point23, point24, point25, point26, point27, point28, point29};
    GeometryVector points3{point31, point32, point33, point34, point35, point36, point37, point38, point39};
    GeometryVector points4{point41, point42, point43, point44, point45, point46, point47, point48, point49};
    Vector1d weight1(1), weight2(0.883022221559489);
    WeightVector weights1{weight1, weight1, weight1, weight2, weight2, weight2, weight2, weight2, weight2};
    WeightVector weights2{weight1, weight1, weight1, weight2, weight2, weight2, weight2, weight2, weight2};
    WeightVector weights3{weight2, weight2, weight2, weight2, weight2, weight2, weight1, weight1, weight1};
    WeightVector weights4{weight2, weight2, weight2, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1, weights1);
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2, weights2);
    auto domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3, weights3);
    auto domain4 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4, weights4);

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
        return vector<double>{0, 0, -90};
    };
    DofMapper dof;
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), 3 * i->GetDomain()->GetDof());
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            cells[i]->Match(cells[j]);
        }
    }
    for (auto &i : cells)
    {
        i->PrintEdgeInfo();
    }
    vector<Triplet<double>> constraint;
    ConstraintAssembler<double> constraint_assemble(dof);
    SparseMatrix<double> constraint_matrix(constraint_assemble.Assemble(cells, constraint), dof.TotalDof());
    constraint_matrix.setFromTriplets(constraint.begin(), constraint.end());

    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(constraint_matrix);
    assert(qr.info() == Success);
    // Extract Q matrix
    SparseMatrix<double> left = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).pruned(1e-13);
    SparseMatrix<double> right = qr.matrixR().topRightCorner(qr.rank(), constraint_matrix.cols() - qr.rank()).pruned(1e-13);

    BiCGSTAB<SparseMatrix<double>> solver;
    solver.compute(left);
    MatrixXd x = solver.solve(-right);
    cout << x << endl;
    cout << qr.rank() << endl;
    x.conservativeResize(x.rows() + x.cols(), x.cols());
    x.bottomLeftCorner(x.cols(), x.cols()) = MatrixXd::Identity(x.cols(), x.cols());
    cout << constraint_matrix * qr.colsPermutation() * x << endl;

    // ofstream myfile1, myfile2, myfile3, myfile4;
    // myfile1.open("domain1.txt");
    // myfile2.open("domain2.txt");
    // myfile3.open("domain3.txt");
    // myfile4.open("domain4.txt");
    // Vector2d u;
    // for (int i = 0; i < 51; i++)
    // {
    //     for (int j = 0; j < 51; j++)
    //     {
    //         u << 1.0 * i / 50, 1.0 * j / 50;
    //         VectorXd position = domain1->AffineMap(u);
    //         myfile1 << position(0) << " " << position(1) << " " << position(2) << endl;
    //         position = domain2->AffineMap(u);
    //         myfile2 << position(0) << " " << position(1) << " " << position(2) << endl;
    //         position = domain3->AffineMap(u);
    //         myfile3 << position(0) << " " << position(1) << " " << position(2) << endl;
    //         position = domain4->AffineMap(u);
    //         myfile4 << position(0) << " " << position(1) << " " << position(2) << endl;
    //     }
    // }
    // myfile1.close();
    // myfile2.close();
    // myfile3.close();
    // myfile4.close();
    return 0;
}