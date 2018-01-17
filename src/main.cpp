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
#include <fstream>
#include <time.h>
#include <boost/math/constants/constants.hpp>

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
    double rad = 40.0 / 180 * boost::math::constants::pi<double>();
    double a = sin(rad) * R;
    double b = a * tan(rad);
    Vector3d point1(-a, 0, 0), point2(-a, L / 2, 0), point3(-a, L, 0), point4(0, 0, b), point5(0, L / 2, b), point6(0, L, b), point7(a, 0, 0), point8(a, L / 2, 0), point9(a, L, 0);
    GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(sin(boost::math::constants::pi<double>() / 2 - rad));
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points, weights);
    int degree, refine;
    cin >> degree >> refine;
    domain->DegreeElevate(degree);
    domain->UniformRefine(refine);
    auto cell = make_shared<Surface<3, double>>(domain);
    cell->SurfaceInitialize();
    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0, -90};
    };
    MembraneStiffnessVisitor<double> membrane_stiffness(body_force);
    BendingStiffnessVisitor<double> bending_stiffness(body_force);
    cell->Accept(membrane_stiffness);
    cell->Accept(bending_stiffness);
    SparseMatrix<double> sparse_bstiffness_triangle_view, rhs;
    SparseMatrix<double> sparse_mstiffness_triangle_view;
    bending_stiffness.StiffnessAssembler(sparse_bstiffness_triangle_view);
    membrane_stiffness.StiffnessAssembler(sparse_mstiffness_triangle_view);
    SparseMatrix<double> sparse_stiffness = (sparse_bstiffness_triangle_view + sparse_mstiffness_triangle_view).template selfadjointView<Eigen::Upper>();
    membrane_stiffness.LoadAssembler(rhs);
    auto south_indices = cell->EdgePointerGetter(0)->Indices(0);
    auto north_indices = cell->EdgePointerGetter(2)->Indices(0);
    vector<int> dirichlet_indices;
    for (const auto &i : *south_indices)
    {
        dirichlet_indices.push_back(3 * i);
    }

    dirichlet_indices.push_back(3 * *(south_indices->begin()) + 1);

    for (const auto &i : *south_indices)
    {
        dirichlet_indices.push_back(3 * i + 2);
    }
    for (const auto &i : *north_indices)
    {
        dirichlet_indices.push_back(3 * i);
    }

    for (const auto &i : *north_indices)
    {
        dirichlet_indices.push_back(3 * i + 2);
    }
    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    MatrixXd global_to_free = MatrixXd::Identity(3 * (domain->GetDof()), 3 * (domain->GetDof()));
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    {
        Accessory::removeRow(global_to_free, *it);
    }
    MatrixXd stiffness_matrix = global_to_free * sparse_stiffness * global_to_free.transpose();
    VectorXd load_vector = global_to_free * rhs;
    VectorXd solution = global_to_free.transpose() * stiffness_matrix.partialPivLu().solve(load_vector);
    GeometryVector solution_ctrl_pts;
    for (int i = 0; i < domain->GetDof(); i++)
    {
        Vector3d temp;
        temp << solution(3 * i + 0), solution(3 * i + 1), solution(3 * i + 2);
        solution_ctrl_pts.push_back(temp);
    }
    auto solution_domain = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domain->KnotVectorGetter(0), domain->KnotVectorGetter(1)}, solution_ctrl_pts, domain->WeightVectorGetter());
    Vector2d u;
    u << 1, .5;
    cout <<setprecision(10)<< solution_domain->AffineMap(u);
    return 0;
}