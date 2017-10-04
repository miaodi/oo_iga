#include <iostream>
#include <eigen3/Eigen/Dense>
#include "PhyTensorNURBSBasis.h"
#include "Topology.h"


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
    a.InitClosed(2, 0, 1);
    Vector2d point1(0,0), point2(0,1), point3(1,1);
    vector<Vector2d> point{point1,point2,point3};
    Vector1d weight1(1), weight2(1.0/sqrt(2)), weight3(1);
    vector<Vector1d> weight{weight1,weight2,weight3};
    auto domain1 = make_shared<PhyTensorNURBSBasis<1,2,double>>(vector<KnotVector<double>>{a},point,weight,true);
    Vector1d pt(.5);
    auto eval = domain1->EvalDerAllTensor(pt,0);
    Vector2d position;
    position.setZero();
    int k=0;
    for(auto i:*eval){
        for(auto j:i.second){
            position+=j*point[k];

        }
        k++;
    }
    cout<<position;
    return 0;
}