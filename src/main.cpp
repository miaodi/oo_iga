#include <iostream>
#include <eigen3/Eigen/Dense>

#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

#include "KnotVector.h"
#include "QuadratureRule.h"
#include "Utility.hpp"

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;

int main()
{
    double b = 1;
    KnotVector<double> a;

    a.InitClosed(1, 0, 1);
    a.Insert(b / 3);
    a.UniformRefine(5);
    QuadratureRule<double> c;
    c.SetUpQuadrature(11);
    c.PrintCurrentQuadrature();
    // BsplineBasis<double> d(a);
    // auto eval = d.EvalDerAll(double(.5), 0);
    // for (auto &i : *eval)
    // {
    //     cout << setprecision(100) << i.second[0] << endl;
    // }
    return 0;
}