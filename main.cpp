#include <iostream>
#include<eigen3/Eigen/Dense>
#include "KnotVector.h"
#include "BsplineBasis.h"
#include "TensorBsplineBasis.h"
#include "PhyTensorBsplineBasis.h"
#include "MultiArray.h"

using namespace Eigen;
using namespace std;
using namespace Accessory;

int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    KnotVector<double> b;
    b.InitClosed(1, 0, 1);
    Vector2d point1(0, 0);
    Vector2d point2(0, 1);
    Vector2d point3(1, 0);
    Vector2d point4(1, 1);

    vector<Vector2d> points({point1, point2, point3, point4});
    PhyTensorBsplineBasis<2, 2, double> domain(a, b, points);
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
    auto x=PartialDerPattern<3>(2);
    for(auto it = x->begin();it!=x->end();++it){
        for(auto number:*it)
            cout<<number<<" ";
        cout<<endl;
    }
    return 0;
}