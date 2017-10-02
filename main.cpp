#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorBsplineBasis.h"
#include "Topology.h"
#include "PostProcess.h"
#include <fstream>
#include <iomanip>
#include <ctime>

using namespace Eigen;
using namespace std;
using namespace Accessory;
using Coordinate=Element<double>::Coordinate;
using CoordinatePairList=Element<double>::CoordinatePairList;
using Quadrature = QuadratureRule<double>::Quadrature;
using QuadList = QuadratureRule<double>::QuadList;
using LoadFunctor = Element<double>::LoadFunctor;
using Vector1d = Matrix<double, 1, 1>;

int main() {
    KnotVector<double> a;
    a.InitClosed(1, 0, 1);
    auto domain = make_shared<TensorBsplineBasis<2>>(a,a);
    Vector2d point(0,0);
    auto val = domain->EvalDerAllTensor(point,3);
    for(auto i:*val){
        for(auto j:i.second){
            cout<<j<<" ";
        }
        cout<<endl;
    }
    BsplineBasis<double> domain1(a);
    auto val1 = domain1.EvalDerAll(0,3);
    for(auto i:*val1){
        for(auto j:i.second){
            cout<<j<<" ";
        }
        cout<<endl;
    }
    return 0;
}