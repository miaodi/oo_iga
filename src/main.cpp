#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "MembraneStiffnessVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "BendingStiffnessVisitor.hpp"
#include "PostProcess.h"
#include "L2StiffnessVisitor.hpp"
#include "DofMapper.hpp"
#include <fstream>
#include <time.h>
#include <boost/math/constants/constants.hpp>
#include "BiharmonicInterfaceVisitor.hpp"
#include "StiffnessAssembler.hpp"
#include "ConstraintAssembler.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <boost/math/special_functions/legendre.hpp>

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

    Vector2d point1(0, 0), point2(0, 1), point3(.4, 0), point4(.4, 1), point5(1, 0), point6(1, 1);

    GeometryVector points1{point1, point2, point3, point4};
    GeometryVector points2{point3, point4, point5, point6};

    int degree, refine;
    cin >> degree >> refine;
    for (int d = 1; d < degree; d++)
    {
        for (int r = 1; r < refine; r++)
        {
            array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 2> domains;
            domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1);
            domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2);

            domains[0]->DegreeElevate(1);
            domains[1]->DegreeElevate(1);

            Vector2d modify1, modify2, modify3;
            modify1 << .5, .4;
            modify2 << .75, .45;
            modify3 << .25, .45;

            domains[0]->CtrPtsSetter(7, modify1);
            domains[0]->CtrPtsSetter(4, modify3);
            domains[1]->CtrPtsSetter(1, modify1);
            domains[1]->CtrPtsSetter(4, modify2);

            for (auto &i : domains)
            {
                i->DegreeElevate(d - 1);
            }
            domains[1]->KnotsInsertion(0, {1.0 / 2});
            domains[1]->KnotsInsertion(1, {1.0 / 3, 2.0 / 3});
            domains[0]->KnotsInsertion(1, {1.0 / 2});

            for (auto &i : domains)
            {
                i->UniformRefine(r);
            }

            vector<shared_ptr<Surface<2, double>>> cells;
            for (int i = 0; i < 2; i++)
            {
                cells.push_back(make_shared<Surface<2, double>>(domains[i]));
                cells[i]->SurfaceInitialize();
            }

            for (int i = 0; i < 1; i++)
            {
                for (int j = i + 1; j < 2; j++)
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
            // MatrixXd dense_constraint = MatrixXd(constraint);
            // FullPivLU<MatrixXd> lu_decomp(dense_constraint);
            // MatrixXd kernel = lu_decomp.kernel();
            // SparseMatrix<double> sp = kernel.sparseView(1e-15);

            // ofstream myfile;
            // myfile.open("example.txt");
            // myfile << dense_constraint;
            // myfile.close();

            SparseMatrix<double> sp1;
            constraint_assemble.AssembleByReducedKernel(sp1);
            // cout << MatrixXd(constraint * sp1).norm() << endl;
            // cout << MatrixXd(constraint * sp).norm() << endl;
            // cout << sp1.cols() << " " << sp.cols() << endl;

            const double pi = 3.141592653589793238462;
            function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{-324 * pow(pi, 4) * (cos(6 * pi * x) - 2 * cos(6 * pi * (x - y)) + cos(6 * pi * y) - 2 * cos(6 * pi * (x + y)))};
            };

            function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{pow(sin(3 * pi * x) * sin(3 * pi * y), 2),
                                      3 * pi * sin(6 * pi * x) * pow(sin(3 * pi * y), 2),
                                      3 * pi * sin(6 * pi * y) * pow(sin(3 * pi * x), 2),
                                      18 * pi * pi * cos(6 * pi * x) * pow(sin(3 * pi * y), 2),
                                      9 * pi * pi * sin(6 * pi * x) * sin(6 * pi * y),
                                      18 * pi * pi * cos(6 * pi * y) * pow(sin(3 * pi * x), 2)};
            };

            StiffnessAssembler<BiharmonicStiffnessVisitor<double>> stiffness_assemble(dof);
            SparseMatrix<double> stiffness_matrix, load_vector;
            stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
            SparseMatrix<double> constrained_stiffness_matrix = sp1.transpose() * stiffness_matrix * sp1;
            SparseMatrix<double> constrained_rhs = sp1.transpose() * load_vector;
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute(constrained_stiffness_matrix);
            VectorXd Solution = sp1 * cg.solve(constrained_rhs);

            vector<KnotVector<double>> solutionDomain1, solutionDomain2;
            solutionDomain1.push_back(domains[0]->KnotVectorGetter(0));
            solutionDomain1.push_back(domains[0]->KnotVectorGetter(1));
            solutionDomain2.push_back(domains[1]->KnotVectorGetter(0));
            solutionDomain2.push_back(domains[1]->KnotVectorGetter(1));
            VectorXd controlDomain1 = Solution.segment(dof.StartingDof(cells[0]->GetID()), domains[0]->GetDof());
            VectorXd controlDomain2 = Solution.segment(dof.StartingDof(cells[1]->GetID()), domains[1]->GetDof());
            vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions(2);
            solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain1, controlDomain1);
            solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>(solutionDomain2, controlDomain2);

            PostProcess<double> post_process(cells, solutions, analytical_solution);
            cout << post_process.RelativeL2Error() << "   " << post_process.RelativeH2Error() << endl;
        }
        cout << endl;
    }
    return 0;
}