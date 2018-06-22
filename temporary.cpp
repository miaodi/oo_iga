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
#include "BiharmonicInterfaceH1.h"
#include "PostProcess.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;

int main()
{
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0), point2(0, 1), point3(.5, 0), point4(.5, 1), point5(1, 0), point6(1, 1);

    GeometryVector point{point1, point2, point3, point4};
    GeometryVector pointt{point3, point4, point5, point6};

    int degree, refine;
    cin >> degree >> refine;
    for (int i = 1; i < degree; i++)
    {
        for (int j = 1; j < refine; j++)
        {
            auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, point);
            auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointt);
            domain1->DegreeElevate(1);
            domain2->DegreeElevate(1);
            Vector2d modify_point_l(.2, .475);
            Vector2d modify_point(.4, .45);
            Vector2d modify_point_r(.7, .475);
            domain1->CtrPtsSetter(4, modify_point_l);
            domain1->CtrPtsSetter(7, modify_point);
            domain2->CtrPtsSetter(1, modify_point);
            domain2->CtrPtsSetter(4, modify_point_r);

            domain1->DegreeElevate(i - 1);
            domain2->DegreeElevate(i - 1);

            domain2->KnotInsertion(0, 1.0 / 3);
            domain2->KnotInsertion(0, 2.0 / 3);
            domain2->KnotInsertion(1, 1.0 / 3);
            domain2->KnotInsertion(1, 2.0 / 3);
            domain1->UniformRefine(j + 1, 1);
            domain2->UniformRefine(j, 1);
            array<shared_ptr<Surface<2, double>>, 2> cells;
            cells[0] = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, false, true, true});
            cells[0]->SurfaceInitialize();
            cells[1] = make_shared<Surface<2, double>>(domain2, array<bool, 4>{true, true, true, false});
            cells[1]->SurfaceInitialize();
            for (int i = 0; i < 1; i++)
            {
                for (int j = i + 1; j < 2; j++)
                    cells[i]->Match(cells[j]);
            }
            const double pi = 3.141592653589793238462;
            function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{-64 * pow(pi, 4) * (cos(4 * pi * x) - 2 * cos(4 * pi * (x - y)) + cos(4 * pi * y) - 2 * cos(4 * pi * (x + y)))};
            };

            function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
                double x = u(0);
                double y = u(1);
                return vector<double>{pow(sin(2 * pi * x) * sin(2 * pi * y), 2)};
            };

            DofMapper<2, double> dof_map;
            BiharmonicMapper<2, double> mapper(dof_map);

            for (int i = 0; i < 2; i++)
            {
                cells[i]->Accept(mapper);
            }

            BiharmonicStiffnessVisitor<2, double> stiffness(dof_map, body_force);
            for (int i = 0; i < 2; i++)
            {
                cells[i]->Accept(stiffness);
            }
            SparseMatrix<double> stiffness_matrix_triangle_view, load_vector;
            stiffness.StiffnessAssembler(stiffness_matrix_triangle_view);
            stiffness.LoadAssembler(load_vector);
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_triangle_view.template selfadjointView<Eigen::Upper>();
            // BiharmonicInterfaceH1<2, double> interface(dof_map);
            BiharmonicInterface<2, double> interface1(dof_map);
            for (int i = 0; i < 2; i++)
            {
                // cells[i]->EdgeAccept(interface);
                cells[i]->EdgeAccept(interface1);
            }
            MatrixXd constraint, constraint1;
            // interface.ConstraintMatrix(constraint);
            interface1.ConstraintMatrix(constraint1);

            MatrixXd global_to_free = MatrixXd::Identity(dof_map.Dof(), dof_map.Dof());
            auto dirichlet_indices = dof_map.GlobalDirichletIndices();
            sort(dirichlet_indices.begin(), dirichlet_indices.end());
            for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); it++)
            {
                Accessory::removeRow(global_to_free, *it);
            }
            SparseMatrix<double> free_stiffness = SparseMatrix<double>(global_to_free.sparseView()) * stiffness_matrix * SparseMatrix<double>(global_to_free.transpose().sparseView());
            VectorXd free_load = global_to_free * load_vector;
            MatrixXd free_constraint = constraint1 * global_to_free.transpose();
            FullPivLU<MatrixXd> lu(free_constraint);
            lu.setThreshold(1e-11);
            MatrixXd ker = lu.kernel();
            SparseMatrix<double> ker_sparse = ker.sparseView();

            SparseMatrix<double> stiffness_sol = ker_sparse.transpose() * free_stiffness * ker_sparse;
            VectorXd load_sol = ker_sparse.transpose() * free_load;
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute(stiffness_sol);
            VectorXd Solution = cg.solve(load_sol);

            VectorXd solution = global_to_free.transpose() * ker_sparse * Solution;

            PostProcess<2, double> post_process(dof_map, solution, analytical_solution);
            for (int i = 0; i < 2; i++)
            {
                cells[i]->Accept(post_process);
            }
            cout << post_process.L2Norm() << endl;
        }
        cout << endl;
    }
    return 0;
}