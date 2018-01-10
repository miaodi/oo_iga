#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "Elasticity2DDeviatoricStiffnessVisitor.hpp"
#include "PressureProjectionVisitor.hpp"
#include "PressureStiffnessVisitor.hpp"
#include "PressureStiffnessDualVisitor.hpp"
#include "PostProcess.hpp"
#include "H1DomainSemiNormVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, double>::WeightVector;
using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
using VectorXmpf = Matrix<mpf_float_100, Dynamic, 1>;
using MatrixXmpf = Matrix<mpf_float_100, Dynamic, Dynamic>;
using Vector1d = Matrix<double, 1, 1>;

int main()
{
    double nu = 0.4999;
    double E = 240.565e6;
    double lambda = nu * E / (1 + nu) / (1 - 2 * nu);
    double mu = E / 2 / (1 + nu);
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0), point2(0, 44), point3(48, 44), point4(48, 60);
    GeometryVector points{point1, point2, point3, point4};
    Vector1d weight1(1);
    WeightVector weights{weight1, weight1, weight1, weight1};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights);
    int degree, refine;
    cin >> degree >> refine;
    domain->DegreeElevate(degree);
    domain->UniformRefine(refine);
    auto cell = make_shared<Surface<2, double>>(domain);
    cell->SurfaceInitialize();

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0};
    };
    function<vector<double>(const VectorXd &)> stress_solution = [](const VectorXd &u) {
        double f = 0;
        if (abs(u(0) - 48)<1e-5)
        {
            f = 6.25e6;
        }
        cout<<u.transpose()<<endl;
        return vector<double>{0, f, 0};
    };

    function<vector<double>(const VectorXd &)> displacement_solution = [](const VectorXd &u) {

        return vector<double>{1, 1};
    };

    Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);
    PressureProjectionVisitor<double> bezier_projection(true), projection(false);
    PressureStiffnessVisitor<double> pressure;
    NeumannBoundaryVisitor<double> neumann(stress_solution);
    cell->Accept(stiffness);
    cell->Accept(bezier_projection);
    cell->Accept(projection);
    cell->Accept(pressure);
    cell->EdgePointerGetter(1)->Accept(neumann);
    SparseMatrix<double> sparse_stiffness_triangle_view, sparse_bezier_projection, sparse_projection, sparse_pressure, sparse_dual_pressure, sparse_h1, rhs;
    stiffness.StiffnessAssembler(sparse_stiffness_triangle_view);
    projection.InnerProductAssembler(sparse_projection);
    bezier_projection.InnerProductAssembler(sparse_bezier_projection);
    pressure.InnerProductAssembler(sparse_pressure);
    MatrixXd global_projection = MatrixXd(sparse_pressure).partialPivLu().solve(MatrixXd(sparse_projection));
    neumann.NeumannBoundaryAssembler(rhs);
    cout<<rhs<<endl;

    auto west_indices = cell->EdgePointerGetter(3)->Indices(0);
    vector<int> dirichlet_indices;
    for (const auto &i : *west_indices)
    {
        dirichlet_indices.push_back(2 * i);
    }
    for (const auto &i : *west_indices)
    {
        dirichlet_indices.push_back(2 * i + 1);
    }
    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    MatrixXd global_to_free = MatrixXd::Identity(2 * (domain->GetDof()), 2 * (domain->GetDof()));
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    {
        Accessory::removeRow(global_to_free, *it);
    }
    SparseMatrix<double> sparse_stiffness = sparse_stiffness_triangle_view.template selfadjointView<Eigen::Upper>();

    MatrixXd stiffness_matrix = global_to_free * ((lambda + 2.0 / 3 * mu) * sparse_projection.transpose() * sparse_bezier_projection + sparse_stiffness) * global_to_free.transpose();
    VectorXd load_vector = global_to_free * rhs;
    VectorXd solution = global_to_free.transpose() * stiffness_matrix.partialPivLu().solve(load_vector);
    cout<<solution;
    return 0;
}