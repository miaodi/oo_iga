#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "DofMapper.hpp"
#include "PhyTensorNURBSBasis.h"
#include "Elasticity2DDeviatoricStiffnessVisitor.hpp"
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
    Vector1d weight1(1), weight2(1.0 / sqrt(2.0)), weight3(1);
    WeightVector weights{weight1, weight2, weight3, weight1, weight2, weight3, weight1, weight2, weight3};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights, false);

    auto cell = make_shared<Surface<2, double>>(domain);
    cell->SurfaceInitialize();

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0};
    };
    Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);

    cell->Accept(stiffness);
    SparseMatrix<double> sparse_stiffness;
    stiffness.StiffnessAssembler(sparse_stiffness);
    cout << MatrixXd(sparse_stiffness);
    return 0;
}