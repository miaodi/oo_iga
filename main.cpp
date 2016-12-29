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
    KnotVector<double> c({0, 0, 0, 0,.2,.4,.4,.7, 1, 1, 1, 1});

    BsplineBasis<double> l(b);
    BsplineBasis<double> m(c);

    vector<KnotVector<double>> n={b,c};
    TensorBsplineBasis<2> k(n);
    TensorBsplineBasis<2> g;
    VectorXd u(2);
    u<<.8,.8;


    auto f = TensorBsplineBasis<2>::PartialDerPattern(2);
    for(auto it1=f->begin();it1!=f->end();++it1){
        for(auto it2= it1->begin();it2!=it1->end();++it2)
            cout<<*it2<<" ";
        cout<<endl;
    }
    auto kkkk = k.EvalTensor(u,2);
    for(auto it=kkkk->begin();it!=kkkk->end();++it) {
        cout << (*it).first <<endl;
        for(auto it_order=(*it).second.begin();it_order!=(*it).second.end();++it_order){
            for(auto it_it=it_order->begin();it_it!=it_order->end();++it_it)
                cout<<*it_it<<" ";
            cout<<endl;
        }
        cout<<endl<<endl;
    }


    auto fff=k.TensorIndex(15);
    for(auto it=fff.begin();it!=fff.end();++it)
        cout<<*it<<" ";
    cout<<endl;
    cout<<k.Index({2,7})<<endl;
    vector<VectorXd> ff(24,u);
    PhyTensorBsplineBasis<2> hahaming(n,ff);
    return 0;
}