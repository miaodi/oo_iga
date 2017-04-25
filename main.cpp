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
    BsplineBasis<double>::BasisFunValPac_ptr result=l.Eval(.00001,4);
    for( auto i=result->begin();i!=result->end();++i){
        cout<<i->first<<" "<<i->second<<endl;
    }
    return 0;
}