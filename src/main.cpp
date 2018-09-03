#include "BiharmonicInterfaceVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "BoundaryAssembler.hpp"
#include "ConstraintAssembler.hpp"
#include "DofMapper.hpp"
#include "L2StiffnessVisitor.hpp"
#include "MembraneStiffnessVisitor.hpp"
#include "PhyTensorNURBSBasis.h"
#include "PostProcess.h"
#include "StiffnessAssembler.hpp"
#include "Surface.hpp"
#include "Utility.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 2, long double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 2, long double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, long double>::WeightVector;
using Vector1ld = Matrix<long double, 1, 1>;

using Vector2ld = Matrix<long double, 2, 1>;
using VectorXld = Matrix<long double, Dynamic, 1>;
using MatrixXld = Matrix<long double, Dynamic, Dynamic>;

const long double Pi = 3.14159265358979323846264338327;

int main()
{
    KnotVector<long double> knot_vector;
    knot_vector.InitClosed( 3, 0, 1 );

    knot_vector.Insert( .1 );
    knot_vector.Insert( .2 );
    knot_vector.Insert( .3 );
    knot_vector.Insert( .4 );
    knot_vector.Insert( .5 );
    knot_vector.Insert( .6 );
    knot_vector.Insert( .7 );
    knot_vector.Insert( .8 );
    knot_vector.Insert( .9 );

    knot_vector.printKnotVector();
    BsplineBasis<long double> basis( knot_vector );
    basis.BezierDualInitialize();
    long double u = 0;
    ofstream file( "ex.txt" );
    file << "X ";
    for ( int j = 0; j < knot_vector.GetDOF(); ++j )
    {
        file << "N" << j << " ";
    }
    file << endl;
    for ( int i = 0; i <= 1000; ++i )
    {
        u = 1. * i / 1000;
        auto evals = basis.EvalDerAll( u, 0 );
        file << u << " ";
        vector<long double> temp( knot_vector.GetDOF(), 0 );
        for ( auto i : *evals )
        {
            temp[i.first] = i.second[0];
        }
        for ( auto i : temp )
        {
            file << i << " ";
        }
        file << endl;
    }

    return 0;
}