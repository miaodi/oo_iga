
#include <iostream>
#include <eigen3/Eigen/Dense>

#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

#include "KnotVector.h"
#include "QuadratureRule.h"
#include "Utility.hpp"
#include "PhyTensorBsplineBasis.h"

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;

using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
int main()
{

    KnotVector<mpf_float_100> a, b;
    a.InitClosed(1, 0, 1);
    Vector2mpf point1(mpf_float_100("0"), mpf_float_100("0"));
    Vector2mpf point2(mpf_float_100("0"), mpf_float_100("2"));
    Vector2mpf point3(mpf_float_100("2"), mpf_float_100("0"));
    Vector2mpf point4(mpf_float_100("2"), mpf_float_100("2"));


    vector<Vector2mpf> points1({point1, point2, point3, point4});
    auto domain1 = make_shared<PhyTensorBsplineBasis<2, 2, mpf_float_100>>(a, a, points1);
    domain1->DegreeElevate(2);
    Vector2mpf u(mpf_float_100("0"), mpf_float_100("0"));
    domain1->UniformRefineDof(50);
    auto eval = domain1->Eval2PhyDerAllTensor(u);
    for (auto i : *eval)
    {
        cout << setprecision(100) << i.second[0] << ", " << i.second[1] << ", " << i.second[2] << endl;
    }
    cout<<domain1->GetDof();
    return 0;
}