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

    BsplineBasis<double> m(a);
    cout<<m.EvalSingle(.1,2,2)<<endl;
    cout<<*(m.Eval(.1,2))<<endl;
    cout<<m.support(0)(1)<<endl;
    vector<KnotVector<double>> n={a,b,c};
    TensorBsplineBasis<3> k(n);

    VectorXd u(3);
    u<<.2,.4,.6;
    cout<<k.EvalSingle(u,2,{0,0,0})<<endl;
    auto f = TensorBsplineBasis<3>::PartialDerPattern(2);
    for(auto it1=f->begin();it1!=f->end();++it1){
        for(auto it2= it1->begin();it2!=it1->end();++it2)
            cout<<*it2<<" ";
        cout<<endl;
    }
    return 0;
}