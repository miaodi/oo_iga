#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "MembraneStiffnessVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
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
#include "BiharmonicStiffnessAssembler.hpp"
#include "ConstraintAssembler.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 2, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

const double Pi = 3.14159265358979323846264338327;

int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(1, 0, 1);

    Vector2d point1(0, 0), point2(.5, 1.5), point3(1.0, 3.0), point4(1.5, 0), point5(1.6, 1.1), point6(2, 1.5), point7(3, 0);

    GeometryVector points1{point1, point2, point4, point5};
    GeometryVector points2{point2, point3, point5, point6};
    GeometryVector points3{point4, point5, point7, point6};

    array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 3> domains;
    domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1);
    domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2);
    domains[2] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3);

    int degree, refine;
    cin >> degree >> refine;
    for (auto &i : domains)
    {
        i->DegreeElevate(degree);
    }
    domains[0]->KnotsInsertion(0, {1.0 / 3, 2.0 / 3});
    domains[0]->KnotsInsertion(1, {1.0 / 3, 2.0 / 3});
    domains[1]->KnotsInsertion(0, {1.0 / 2});
    domains[1]->KnotsInsertion(1, {1.0 / 2});
    domains[2]->KnotsInsertion(0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5});
    domains[2]->KnotsInsertion(1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5});

    for (auto &i : domains)
    {
        i->UniformRefine(refine);
    }

    vector<shared_ptr<Surface<2, double>>> cells;
    for (int i = 0; i < 3; i++)
    {
        cells.push_back(make_shared<Surface<2, double>>(domains[i]));
        cells[i]->SurfaceInitialize();
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i + 1; j < 3; j++)
        {
            cells[i]->Match(cells[j]);
        }
    }
    DofMapper dof;
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), i->GetDomain()->GetDof());
    }
    ConstraintAssembler<2, 2, double> constraint_assemble(dof);
    constraint_assemble.ConstraintCreator(cells);
    SparseMatrix<double, RowMajor> constraint;
    constraint_assemble.AssembleConstraint(constraint);
    SparseMatrix<double> sp;
    constraint_assemble.AssembleByReducedKernel(sp);

    cout << (constraint * sp).norm() << endl;
    MatrixXd dense_constraint = constraint;
    cout << dense_constraint.rows() << " " << dense_constraint.cols() << endl;
    FullPivLU<MatrixXd> lu_decomp(dense_constraint);
    MatrixXd kernel = lu_decomp.kernel();

    cout << (constraint * kernel).norm() << endl;

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{pow(x, 2) * pow(3 * x - y, 2) * pow(-9 + 3 * x + 2 * y, 2) * sin(2 * Pi * x) * sin(2 * Pi * y)};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{8 * (-4 * Pi * cos(2 * Pi * x) * (-2 * Pi * x * (135 * pow(x, 3) - 108 * pow(x, 2) * y - 27 * x * (27 - 18 * y + 2 * pow(y, 2)) + 2 * y * (81 - 54 * y + 8 * pow(y, 2))) * cos(2 * Pi * y) + (972 * pow(Pi, 2) * pow(x, 5) + 540 * pow(Pi, 2) * pow(x, 4) * (-9 + y) + 9 * y * (81 - 27 * y + 2 * pow(y, 2)) - 9 * pow(x, 2) * (-783 + 12 * (7 + 27 * pow(Pi, 2)) * y - 108 * pow(Pi, 2) * pow(y, 2) + 8 * pow(Pi, 2) * pow(y, 3)) + x * (-4455 + 108 * y + 6 * (23 + 54 * pow(Pi, 2)) * pow(y, 2) - 144 * pow(Pi, 2) * pow(y, 3) + 16 * pow(Pi, 2) * pow(y, 4)) - 216 * pow(x, 3) * (11 + pow(Pi, 2) * (-27 + pow(y, 2)))) * sin(2 * Pi * y)) +
                                   sin(2 * Pi * x) * (-4 * Pi * (108 * pow(Pi, 2) * pow(x, 5) - 108 * pow(Pi, 2) * pow(x, 4) * y + y * (-81 + 54 * y - 8 * pow(y, 2)) + 27 * x * (27 - 18 * y + 2 * pow(y, 2)) + 2 * pow(x, 2) * (27 + 3 * (23 + 54 * pow(Pi, 2)) * y - 108 * pow(Pi, 2) * pow(y, 2) + 16 * pow(Pi, 2) * pow(y, 3)) - 36 * pow(x, 3) * (7 + pow(Pi, 2) * (27 - 18 * y + 2 * pow(y, 2)))) * cos(2 * Pi * y) +
                                                      (2268 + 648 * pow(Pi, 4) * pow(x, 6) + 432 * pow(Pi, 4) * pow(x, 5) * (-9 + y) -
                                                       108 * y - 3 * (19 + 216 * pow(Pi, 2)) * pow(y, 2) + 288 * pow(Pi, 2) * pow(y, 3) -
                                                       32 * pow(Pi, 2) * pow(y, 4) -
                                                       48 * pow(Pi, 2) * pow(x, 3) *
                                                           (-783 + (84 + 81 * pow(Pi, 2)) * y - 27 * pow(Pi, 2) * pow(y, 2) +
                                                            2 * pow(Pi, 2) * pow(y, 3)) +
                                                       18 * x * (-378 + (39 + 648 * pow(Pi, 2)) * y - 216 * pow(Pi, 2) * pow(y, 2) + 16 * pow(Pi, 2) * pow(y, 3)) -
                                                       216 * pow(Pi, 2) * pow(x, 4) * (44 + pow(Pi, 2) * (-27 + pow(y, 2))) +
                                                       pow(x, 2) * (3495 + 8 * pow(Pi, 4) * pow(9 - 2 * y, 2) * pow(y, 2) +
                                                                    24 * pow(Pi, 2) * (-1485 + 36 * y + 46 * pow(y, 2)))) *
                                                          sin(2 * Pi * y)))};
    };

    // BiharmonicStiffnessAssembler<double> stiffness_assemble(dof);
    // SparseMatrix<double> stiffness_matrix, load_vector;
    // stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
    // ofstream fout0;
    // fout0.open("proposed.txt");
    // fout0 << MatrixXd(sp.transpose() * stiffness_matrix * sp) << endl;
    // ofstream fout1;
    // fout1.open("direct.txt");
    // fout1 << MatrixXd(kernel.transpose() * stiffness_matrix * kernel) << endl;
    // cout << stiffness_matrix.cols() << endl;
    return 0;
}