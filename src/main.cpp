#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "DofMapper.hpp"
#include "PhyTensorNURBSBasis.h"
#include "Elasticity2DDeviatoricStiffnessVisitor.hpp"
#include "PressureProjectionVisitor.hpp"
#include "PressureStiffnessVisitor.hpp"
#include "H1DomainSemiNormVisitor.hpp"
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
    a.Insert(.5);
    auto c = Accessory::BezierExtraction<double>(a);
    // BsplineBasis<double> basis(a);
    // basis.BezierDualInitialize();
    // Vector2d point1(-4, 0), point2(-4, 4), point3(0, 4), point4(-2.5, 0), point5(-2.5, 2.5), point6(0, 2.5), point7(-1, 0), point8(-1, 1), point9(0, 1);
    // for (int i = 0; i < 5; i++)
    // {
    //     GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    //     Vector1d weight1(1), weight2(1.0 / sqrt(2.0)), weight3(1);
    //     WeightVector weights{weight1, weight2, weight3, weight1, weight2, weight3, weight1, weight2, weight3};
    //     auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights, false);
    //     // domain->DegreeElevate(2);
    //     domain->UniformRefine(i);
    //     auto cell = make_shared<Surface<2, double>>(domain);
    //     cell->SurfaceInitialize();

    //     function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
    //         return vector<double>{0, 0};
    //     };
    //     Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);
    //     PressureProjectionVisitor<double> projection;
    //     PressureStiffnessVisitor<double> pressure;
    //     H1DomainSemiNormVisitor<double> h1_norm;
    //     cell->Accept(stiffness);
    //     cell->Accept(projection);
    //     cell->Accept(pressure);
    //     cell->Accept(h1_norm);
    //     SparseMatrix<double> sparse_stiffness, sparse_projection, sparse_pressure, sparse_h1;
    //     stiffness.StiffnessAssembler(sparse_stiffness);
    //     projection.InnerProductAssembler(sparse_projection);
    //     pressure.InnerProductAssembler(sparse_pressure);
    //     h1_norm.InnerProductAssembler(sparse_h1);

    //     // ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    //     // cg.compute(sparse_pressure);
    //     // MatrixXd Solution = cg.solve(sparse_projection);

    //     auto south_indices = cell->EdgePointerGetter(0)->Indices(0);
    //     auto north_indices = cell->EdgePointerGetter(2)->Indices(0);
    //     vector<int> dirichlet_indices;
    //     for (const auto &i : *south_indices)
    //     {
    //         dirichlet_indices.push_back(2 * i + 1);
    //     }
    //     for (const auto &i : *north_indices)
    //     {
    //         dirichlet_indices.push_back(2 * i);
    //     }
    //     sort(dirichlet_indices.begin(), dirichlet_indices.end());
    //     MatrixXd global_to_free = MatrixXd::Identity(2 * (domain->GetDof()), 2 * (domain->GetDof()));
    //     for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    //     {
    //         Accessory::removeRow(global_to_free, *it);
    //     }
    //     MatrixXd h1_solve = global_to_free * sparse_h1 * global_to_free.transpose();
    //     MatrixXd b_solve = global_to_free * sparse_projection.transpose() * sparse_pressure * sparse_projection * global_to_free.transpose();
    //     MatrixXd solution = h1_solve.partialPivLu().solve(b_solve);
    //     VectorXd eigen_value = solution.eigenvalues().array().abs();
    //     sort(eigen_value.data(), eigen_value.data() + eigen_value.size());
    //     cout << eigen_value.transpose() << endl;
    // }
    return 0;
}