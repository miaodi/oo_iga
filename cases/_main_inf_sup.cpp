#include <iostream>
#include <Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "Elasticity2DDeviatoricStiffnessVisitor.hpp"
#include "PressureProjectionVisitor.hpp"
#include "PressureStiffnessVisitor.hpp"
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
    KnotVector<double> a;
    a.InitClosed(2, 0, 1);
    Vector2d point1(-4, 0), point2(-4, 4), point3(0, 4), point4(-2.5, 0), point5(-2.5, 2.5), point6(0, 2.5), point7(-1, 0), point8(-1, 1), point9(0, 1);

    GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(1.0 / sqrt(2.0));
    WeightVector weights{weight1, weight2, weight1, weight1, weight2, weight1, weight1, weight2, weight1};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights, false);
    int degree, refine;
    cin >> degree >> refine;
    domain->DegreeElevate(degree);
    domain->UniformRefine(refine);
    auto cell = make_shared<Surface<2, double>>(domain);
    cell->SurfaceInitialize();

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0};
    };
    const double pi = 3.14159265358979323846264338327;
    function<vector<double>(const VectorXd &)> stress_solution = [&pi](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        double r = sqrt(x * x + y * y);
        double theta = acos(x / r);
        double T = 1e7;
        double sigma_rr, sigma_tt, sigma_rt;
        sigma_rr = T / 2.0 * (1 - pow(1.0 / r, 2)) + T / 2.0 * (1 - 4 * pow(1.0 / r, 2) + 3 * pow(1.0 / r, 4)) * cos(2 * theta);
        sigma_tt = T / 2.0 * (1 + pow(1.0 / r, 2)) - T / 2.0 * (1 + 3 * pow(1.0 / r, 4)) * cos(2 * theta);
        sigma_rt = -T / 2.0 * (1 + 2 * pow(1.0 / r, 2) - 3 * pow(1.0 / r, 4)) * sin(2 * theta);
        MatrixXd stress_tensor_polar(2, 2), stress_tensor_cartisan(2, 2), transform(2, 2);
        transform << cos(theta), -sin(theta), sin(theta), cos(theta);
        stress_tensor_polar << sigma_rr, sigma_rt, sigma_rt, sigma_tt;
        stress_tensor_cartisan = transform * stress_tensor_polar * transform.transpose();
        return vector<double>{stress_tensor_cartisan(0, 0), stress_tensor_cartisan(1, 1), stress_tensor_cartisan(0, 1)};
    };

    Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);
    PressureProjectionVisitor<double> projection;
    PressureStiffnessVisitor<double> pressure;
    H1DomainSemiNormVisitor<double> h1_norm;
    cell->Accept(stiffness);
    cell->Accept(projection);
    cell->Accept(pressure);
    cell->Accept(h1_norm);
    SparseMatrix<double> sparse_stiffness, sparse_projection, sparse_pressure, sparse_h1, rhs;
    stiffness.StiffnessAssembler(sparse_stiffness);
    projection.InnerProductAssembler(sparse_projection);
    pressure.InnerProductAssembler(sparse_pressure);
    h1_norm.InnerProductAssembler(sparse_h1);

    auto south_indices = cell->EdgePointerGetter(0)->Indices(0);
    auto north_indices = cell->EdgePointerGetter(2)->Indices(0);
    vector<int> dirichlet_indices;
    for (const auto &i : *south_indices)
    {
        dirichlet_indices.push_back(2 * i + 1);
    }
    for (const auto &i : *north_indices)
    {
        dirichlet_indices.push_back(2 * i);
    }
    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    MatrixXd global_to_free = MatrixXd::Identity(2 * (domain->GetDof()), 2 * (domain->GetDof()));
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    {
        Accessory::removeRow(global_to_free, *it);
    }

    MatrixXd h1_solve = global_to_free * sparse_h1 * global_to_free.transpose();
    SparseLU<SparseMatrix<double>> solver;
    solver.analyzePattern(sparse_pressure);
    solver.factorize(sparse_pressure);
    MatrixXd projection_solution = solver.solve(sparse_projection);
    MatrixXd b_solve = global_to_free * sparse_projection.transpose() * projection_solution * global_to_free.transpose();
    MatrixXd solution = h1_solve.partialPivLu().solve(b_solve);
    VectorXd eigen_value = solution.eigenvalues().array().abs();
    sort(eigen_value.data(), eigen_value.data() + eigen_value.size());
    cout << eigen_value.transpose() << endl;

    return 0;
}