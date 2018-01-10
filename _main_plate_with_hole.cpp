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
    double nu = 0.49999;
    double E = 1e11;
    double lambda = nu * E / (1 + nu) / (1 - 2 * nu);
    double mu = E / 2 / (1 + nu);
    KnotVector<double> a;
    a.InitClosed(2, 0, 1);
    Vector2d point1(-1, 0), point2(-2.5, 0), point3(-4, 0), point4(-1, 1), point5(-2.5, 2.5), point6(-4, 4), point7(0, 1), point8(0, 2.5), point9(0, 4);
    GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(1.0 / sqrt(2.0)), weight3(1);
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights);
    int degree, refine;
    cin >> degree >> refine;
    domain->DegreeElevate(degree);
    domain->KnotInsertion(0, .5);
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

    function<vector<double>(const VectorXd &)> displacement_solution = [&pi, &nu, &E](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        double T = 1e7;
        double mu = E / 2 / (1 + nu);
        double a = 1;
        double kappa = 3 - 4 * nu;
        double r = sqrt(pow(x, 2) + pow(y, 2));
        double theta = acos(x / r);
        Vector2d result;
        result(0) = T * a / 8 / mu * (r / a * (kappa + 1) * cos(theta) + 2.0 * a / r * ((1 + kappa) * cos(theta) + cos(3 * theta)) - 2.0 * pow(a / r, 3) * cos(3 * theta));
        result(1) = T * a / 8 / mu * (r / a * (kappa - 3) * sin(theta) + 2.0 * a / r * ((1 - kappa) * sin(theta) + sin(3 * theta)) - 2.0 * pow(a / r, 3) * sin(3 * theta));
        return vector<double>{result(0), result(1)};
    };

    Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);
    PressureProjectionVisitor<double> bezier_projection(true), projection(false);
    PressureStiffnessVisitor<double> pressure;
    NeumannBoundaryVisitor<double> neumann(stress_solution);
    cell->Accept(stiffness);
    cell->Accept(bezier_projection);
    cell->Accept(projection);
    cell->Accept(pressure);
    cell->EdgePointerGetter(2)->Accept(neumann);
    SparseMatrix<double> sparse_stiffness_triangle_view, sparse_bezier_projection, sparse_projection, sparse_pressure, sparse_dual_pressure, sparse_h1, rhs;
    stiffness.StiffnessAssembler(sparse_stiffness_triangle_view);
    projection.InnerProductAssembler(sparse_projection);
    bezier_projection.InnerProductAssembler(sparse_bezier_projection);
    pressure.InnerProductAssembler(sparse_pressure);
    MatrixXd global_projection = MatrixXd(sparse_pressure).partialPivLu().solve(MatrixXd(sparse_projection));
    neumann.NeumannBoundaryAssembler(rhs);

    auto east_indices = cell->EdgePointerGetter(1)->Indices(0);
    auto west_indices = cell->EdgePointerGetter(3)->Indices(0);
    vector<int> dirichlet_indices;
    for (const auto &i : *east_indices)
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
    GeometryVector solution_control_point_vector;
    for (int i = 0; i < solution.size(); i += 2)
    {
        solution_control_point_vector.push_back((VectorXd(2) << solution(i), solution(i + 1)).finished());
    }
    VectorXd pressure_solution = sparse_bezier_projection * solution;
    WeightVector pressure_point_vector;
    for (int i = 0; i < pressure_solution.size(); i++)
    {
        pressure_point_vector.push_back((VectorXd(1) << pressure_solution(i)).finished());
    }
    vector<KnotVector<double>> solution_knot_vector;
    solution_knot_vector.push_back(domain->KnotVectorGetter(0));
    solution_knot_vector.push_back(domain->KnotVectorGetter(1));
    PhyTensorBsplineBasis<2, 2, double> solution_domain(solution_knot_vector, solution_control_point_vector);
    solution_knot_vector[0].erase(solution_knot_vector[0].begin());
    solution_knot_vector[0].erase(solution_knot_vector[0].end() - 1);
    solution_knot_vector[1].erase(solution_knot_vector[1].begin());
    solution_knot_vector[1].erase(solution_knot_vector[1].end() - 1);
    PhyTensorBsplineBasis<2, 1, double> pressure_domain(solution_knot_vector, pressure_point_vector);
    PostProcess<2, double> post(solution_domain, pressure_domain, displacement_solution, stress_solution);
    cell->Accept(post);
    cout << post.L2Norm() << endl;
    cout << post.L2StressNorm() << endl;
    cout << post.L2EnergyNorm() << endl;
    return 0;
}