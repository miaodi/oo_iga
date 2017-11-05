#include <iostream>
#include <eigen3/Eigen/Dense>

#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

#include "KnotVector.h"
#include "QuadratureRule.h"
#include "Utility.hpp"
#include "BsplineBasis.h"

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;

int main()
{
    mpf_float_100 b = 1;
    KnotVector<mpf_float_100> a;

    a.InitClosed(1, 0, 1);
    a.Insert(b / 3);
    a.UniformRefine(5);
    QuadratureRule<mpf_float_100> c;
    c.SetUpQuadrature(11);
    c.PrintCurrentQuadrature();
    BsplineBasis<mpf_float_100> d(a);
    auto eval = d.EvalDerAll(mpf_float_50(.5), 0);
    for (auto &i : *eval)
    {
        cout << setprecision(100) << i.second[0] << endl;
    }
    return 0;
}