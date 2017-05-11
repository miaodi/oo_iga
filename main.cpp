#include <iostream>
#include<eigen3/Eigen/Dense>
#include "PhyTensorBsplineBasis.h"
#include "MultiArray.h"
#include "MmpMatrix.h"
#include "QuadratureRule.h"
#include "Topology.h"
using namespace Eigen;
using namespace std;
using namespace Accessory;

int main() {
    /*
    MatrixXd test=MatrixXd::Random(8,8);
    test.col(1).setZero();
    test.col(5).setZero();
    test.row(2).setZero();
    test.row(4).setZero();
    cout<<test<<endl<<endl;
    mmpMatrix<double,Dynamic,Dynamic> a(test);
    a.removeZero();
    cout<<a<<endl;
    cout<<test.inverse();
    QuadratureRule<double> hahaming;
    hahaming.SetQuadrature(5);
    */
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    KnotVector<double> b;
    b.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 2);
    Vector2d point3(1, 0);
    Vector2d point4(1, 1);
    vector<Vector2d> points({point1, point2, point3, point4});
    auto domain = make_shared<PhyTensorBsplineBasis<2, 2, double>>(a, b, points);
    Edge<double> left(domain);
    left.GetOrient();
    /*
    cout<<domain.AffineMap(Vector2d(.5,.2))<<endl;
    domain.DegreeElevate(1,2);
    domain.DegreeElevate(0,3);
    cout<<domain.AffineMap(Vector2d(0,1))<<endl;
    domain.UniformRefine(0,3);
    domain.UniformRefine(1,5);
    cout<<domain.AffineMap(Vector2d(.5,.2))<<endl;
    domain.PrintKnots(0);
    domain.PrintKnots(1);
    cout<<domain.Jacobian(Vector2d(.1,.2))<<endl;

    */
        return 0;
}