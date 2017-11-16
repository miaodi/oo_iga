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
#include "PostProcess.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
using VectorXmpf = Matrix<mpf_float_100, Dynamic, 1>;
using MatrixXmpf = Matrix<mpf_float_100, Dynamic, Dynamic>;
int main()
{
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(-sqrt(3), -1), point2(-sqrt(3) / 2, .5), point3(0, 2), point4(0, -1), point5(0, 0), point6(sqrt(3) / 2, .5), point7(sqrt(3), -1);

    GeometryVector point{point1, point2, point4, point5};
    GeometryVector pointt{point2, point3, point5, point6};
    GeometryVector pointtt{point4, point5, point7, point6};

    int degree, refine;
    cin >> degree >> refine;

    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, point);
    auto domain2 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointt);
    auto domain3 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, pointtt);

    domain1->DegreeElevate(degree);
    domain2->DegreeElevate(degree);
    domain3->DegreeElevate(degree);

    domain1->UniformRefine(refine, 1);
    domain2->UniformRefine(refine, 1);
    domain3->UniformRefine(refine, 1);
    array<shared_ptr<Surface<2, double>>, 3> cells;
    cells[0] = make_shared<Surface<2, double>>(domain1, array<bool, 4>{true, false, false, true});
    cells[0]->SurfaceInitialize();
    cells[1] = make_shared<Surface<2, double>>(domain2, array<bool, 4>{false, false, true, true});
    cells[1]->SurfaceInitialize();
    cells[2] = make_shared<Surface<2, double>>(domain3, array<bool, 4>{true, true, false, false});
    cells[2]->SurfaceInitialize();

    for (int i = 0; i < 2; i++)
    {
        for (int j = i + 1; j < 3; j++)
            cells[i]->Match(cells[j]);
    }

    const double pi = 3.14159265358979323846264338327;
    function<vector<double>(const VectorXd &)> body_force = [&pi](const VectorXd &u) {
        return vector<double>{144 * pow(-3 * u(0) + sqrt(3) * (u(1) - 2), 2) + 144 * pow(3 * u(0) + sqrt(3) * (u(1) - 2), 2) + 1728 * pow(u(1) + 1, 2)};
    };

    function<vector<double>(const VectorXd &)> analytical_solution = [&pi](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        return vector<double>{pow((u(1) + 1) * (sqrt(3) * (u(1) - 2) - 3 * u(0)) * (sqrt(3) * (u(1) - 2) + 3 * u(0)), 2),
                              36 * x * (3 * x - sqrt(3) * (y - 2)) * (3 * x + sqrt(3) * (y - 2)) * pow(1 + y, 2),
                              18 * (3 * x - sqrt(3) * (y - 2)) * (3 * x + sqrt(3) * (y - 2)) * (1 + y) * (x * x - (y - 2) * y)};
    };

    DofMapper<2, double> dof_map;
    BiharmonicMapper<2, double> mapper(dof_map);

    for (int i = 0; i < 3; i++)
    {
        cells[i]->Accept(mapper);
    }

    PoissonDirichletBoundaryVisitor<2, double> boundary(dof_map, analytical_solution);
    for (int i = 0; i < 3; i++)
    {
        cells[i]->EdgeAccept(boundary);
    }

    // return 0;
}