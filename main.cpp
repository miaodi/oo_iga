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
    Vector2d point2(0, 2);
    Vector2d point3(1, 0);
    Vector2d point4(1, 1);
    TensorBsplineBasis<2, double> XX(a, a);
    auto receive = XX.EvalDerAllTensor(Vector2d(.2, .4), 3);

    for (auto it =receive->begin();it!=receive->end();++it)
        cout<<it->second[0]<<" ";
    cout<<endl;
    vector<Vector2d> points({point1, point2, point3, point4});
    PhyTensorBsplineBasis<2, 2, double> domain(a, b, points);
    domain.DegreeElevate(0,2);
    domain.DegreeElevate(1,2);
    domain.PrintKnots(0);
    auto compare=domain.EvalDerAllTensor(Vector2d(.2, .4),2);
    auto compare1=domain.Eval2DerAllTensor(Vector2d(.2, .4));
    for(auto& i:(*compare))
        cout<<i.second[4]<<" ";
    cout<<endl;
    for(auto& i:(*compare1))
        cout<<i.second[4]<<" ";
    cout<<endl;
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