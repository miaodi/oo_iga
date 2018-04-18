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

    vector<int> boundary_indices;

    for (auto &i : cells)
    {
        int id = i->GetID();
        int starting_dof = dof.StartingDof(id);
        for (int j = 0; j < 4; j++)
        {
            if (!i->EdgePointerGetter(j)->IsMatched())
            {
                auto local_boundary_indices = i->EdgePointerGetter(j)->Indices(1, 1);
                std::for_each(local_boundary_indices.begin(), local_boundary_indices.end(), [&](int &index) { index += starting_dof; });
                boundary_indices.insert(boundary_indices.end(), local_boundary_indices.begin(), local_boundary_indices.end());
            }
        }
    }
    sort(boundary_indices.begin(), boundary_indices.end());
    boundary_indices.erase(unique(boundary_indices.begin(), boundary_indices.end()), boundary_indices.end());

    ConstraintAssembler<2, 2, double> constraint_assemble(dof);
    constraint_assemble.ConstraintCreator(cells);
    constraint_assemble.Additional_Constraint(boundary_indices);
    SparseMatrix<double, RowMajor> constraint;
    constraint_assemble.AssembleConstraint(constraint);

    for (auto i : boundary_indices)
    {
        constraint.conservativeResize(constraint.rows() + 1, dof.TotalDof());
        SparseVector<double> sparse_vector(dof.TotalDof());
        sparse_vector.coeffRef(i) = 1.0;
        constraint.bottomRows(1) = sparse_vector.transpose();
    }
    MatrixXd dense_constraint = MatrixXd(constraint);
    FullPivLU<MatrixXd> lu_decomp(dense_constraint);
    MatrixXd kernel = lu_decomp.kernel();
    SparseMatrix<double> sp = kernel.sparseView(1e-15);

    // ofstream myfile;
    // myfile.open("example.txt");
    // myfile << dense_constraint;
    // myfile.close();

    SparseMatrix<double> sp1;
    constraint_assemble.AssembleByReducedKernel(sp1);
    cout << MatrixXd(constraint * sp1).norm() << endl;
    cout << MatrixXd(constraint * sp).norm() << endl;
    cout << sp1.cols() << " " << sp.cols() << endl;
    function<vector<double>(const VectorXd &)> analytical_solution = [](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{pow(3 * x - y, 2) * pow(y, 2) * pow(-9 + 3 * x + 2 * y, 2) * sin(2 * Pi * x) * sin(2 * Pi * y)};
    };

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{-96 * Pi * cos(2 * Pi * x) * (2 * Pi * y * (-108 * pow(x, 3) - 81 * pow(x, 2) * (-6 + y) + 18 * x * (-27 + 2 * pow(y, 2)) + y * (243 - 108 * y + 10 * pow(y, 2))) * cos(2 * Pi * y) + (54 * pow(x, 3) * (-1 + 4 * pow(Pi, 2) * pow(y, 2)) + 27 * pow(x, 2) * (9 - 3 * y - 36 * pow(Pi, 2) * pow(y, 2) + 4 * pow(Pi, 2) * pow(y, 3)) + y * (243 + 81 * y - (7 + 324 * pow(Pi, 2)) * pow(y, 2) + 108 * pow(Pi, 2) * pow(y, 3) - 8 * pow(Pi, 2) * pow(y, 4)) - 9 * x * (27 - 12 * (-1 + 9 * pow(Pi, 2)) * pow(y, 2) + 4 * pow(Pi, 2) * pow(y, 4))) * sin(2 * Pi * y)) +
                              8 * sin(2 * Pi * x) * (-12 * Pi * (108 * pow(Pi, 2) * pow(x, 4) * y + 27 * pow(x, 3) * (-1 + 4 * pow(Pi, 2) * (-6 + y) * y) + x * (243 + 162 * y - 3 * (7 + 324 * pow(Pi, 2)) * pow(y, 2) + 432 * pow(Pi, 2) * pow(y, 3) - 40 * pow(Pi, 2) * pow(y, 4)) + y * (-405 + 180 * y + (-22 + 216 * pow(Pi, 2)) * pow(y, 2) - 120 * pow(Pi, 2) * pow(y, 3) + 16 * pow(Pi, 2) * pow(y, 4)) - 36 * pow(x, 2) * y * (3 + pow(Pi, 2) * (-27 + 2 * pow(y, 2)))) * cos(2 * Pi * y) + (972 - 540 * y + (261 - 9720 * pow(Pi, 2)) * pow(y, 2) + 2880 * pow(Pi, 2) * pow(y, 3) + 24 * pow(Pi, 2) * (-11 + 27 * pow(Pi, 2)) * pow(y, 4) - 288 * pow(Pi, 4) * pow(y, 5) + 32 * pow(Pi, 4) * pow(y, 6) + 648 * pow(Pi, 2) * pow(x, 4) * (-1 + pow(Pi, 2) * pow(y, 2)) + 432 * pow(Pi, 2) * pow(x, 3) * (9 - 3 * y - 9 * pow(Pi, 2) * pow(y, 2) + pow(Pi, 2) * pow(y, 3)) - 6 * x * (162 - 3 * (17 + 648 * pow(Pi, 2)) * y - 648 * pow(Pi, 2) * pow(y, 2) + 8 * pow(Pi, 2) * (7 + 81 * pow(Pi, 2)) * pow(y, 3) - 216 * pow(Pi, 4) * pow(y, 4) + 16 * pow(Pi, 4) * pow(y, 5)) - 27 * pow(x, 2) * (-15 + 8 * pow(Pi, 4) * pow(y, 2) * (-27 + pow(y, 2)) + 24 * pow(Pi, 2) * (9 + 4 * pow(y, 2)))) * sin(2 * Pi * y))};
    };

    BiharmonicStiffnessAssembler<double> stiffness_assemble(dof);
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
    SparseMatrix<double> constrained_stiffness_matrix = sp1.transpose() * stiffness_matrix * sp1;
    SparseMatrix<double> constrained_rhs = sp1.transpose() * load_vector;
    ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(constrained_stiffness_matrix);
    VectorXd Solution = sp1 * cg.solve(constrained_rhs);

    vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3;
    solutionDomain1.push_back(domains[0]->KnotVectorGetter(0));
    solutionDomain1.push_back(domains[0]->KnotVectorGetter(1));
    solutionDomain2.push_back(domains[1]->KnotVectorGetter(0));
    solutionDomain2.push_back(domains[1]->KnotVectorGetter(1));
    solutionDomain3.push_back(domains[2]->KnotVectorGetter(0));
    solutionDomain3.push_back(domains[2]->KnotVectorGetter(1));
    VectorXd controlDomain1 = Solution.segment(dof.StartingDof(cells[0]->GetID()), domains[0]->GetDof());
    VectorXd controlDomain2 = Solution.segment(dof.StartingDof(cells[1]->GetID()), domains[1]->GetDof());
    VectorXd controlDomain3 = Solution.segment(dof.StartingDof(cells[2]->GetID()), domains[2]->GetDof());
    vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions(3);
    solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain1, controlDomain1);
    solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain2, controlDomain2);
    solutions[2] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain3, controlDomain3);
    Vector2d u;
    u << .5, .5;
    cout << setprecision(15) << solutions[0]->AffineMap(u) << " " << analytical_solution(domains[0]->AffineMap(u))[0];
    return 0;
}