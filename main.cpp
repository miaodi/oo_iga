#include <iostream>
#include<eigen3/Eigen/Dense>
#include "KnotVector.h"
#include "BsplineBasis.h"
#include "TensorBsplineBasis.h"

using namespace Eigen;
using namespace std;

int main() {
    KnotVector<double> a({0, 0, 0, 1, 1, 1});
    KnotVector<double> b({0, 0, 1, 1});
    KnotVector<double> c({0, 0, 0, 0, 1, 1, 1, 1});

    vector<KnotVector<double>> d = {a, b, c};
    TensorBsplineBasis<3, double> m(d);
    cout<<m.GetDof()<<endl;
    VectorXd u(3);
    u << .09, .5009, .023;
    m.TensorEval(u, 1);
    return 0;
}