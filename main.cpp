#include <iostream>
#include<eigen3/Eigen/Dense>
#include "KnotVector.h"
#include "BsplineBasis.h"
#include "TensorBsplineBasis.h"
#include "PhyTensorBsplineBasis.h"

using namespace Eigen;
using namespace std;

int main() {
    KnotVector<double> a({0, 0, 0, 1, 1, 1});
    KnotVector<double> b({0, 0, .5, 1, 1});
    KnotVector<double> c({0, 0, 0, 0, .2, .4, .4, .7, 1, 1, 1, 1});
    KnotVector<double> d;
    KnotVector<double> e(c.UniKnotUnion(d));


    BsplineBasis<double> l(c);
    BsplineBasis<double> m(c);

    TensorBsplineBasis<2, double> twoDdomain(l, m);
    cout << twoDdomain.GetDof() << endl;
    VectorXd u(2);
    u << 1, 1;
    for (int i = 0; i < 10000; ++i)
        TensorBsplineBasis<2, double>::BasisFunValPac_ptr test = twoDdomain.EvalTensor(u);

    return 0;
}