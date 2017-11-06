
#include <iostream>
#include <eigen3/Eigen/Dense>

#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

#include "KnotVector.h"
#include "QuadratureRule.h"
#include "Utility.hpp"
#include "PhyTensorBsplineBasis.h"

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;

int main()
{

    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 2);
    Vector2d point3(2, -.5);
    Vector2d point4(2, 2.5);
    Vector2d point5(4, 0);
    Vector2d point6(4, 2);

    vector<Vector2d> points1({point1, point2, point3, point4});
    vector<Vector2d> points2({point3, point4, point5, point6});
    Vector2d u(1.0 / 3, .8);
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, a, points1);
    cout << domain1->AffineMap(u) << endl;
    domain1->DegreeElevate(2);
    domain1->UniformRefine(3);
    cout << domain1->AffineMap(u) << endl;
    return 0;
}