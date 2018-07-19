// #include "BiharmonicInterfaceVisitor.hpp"
// #include "BiharmonicStiffnessVisitor.hpp"
// #include "BoundaryAssembler.hpp"
// #include "ConstraintAssembler.hpp"
// #include "DofMapper.hpp"
// #include "L2StiffnessVisitor.hpp"
// #include "MembraneStiffnessVisitor.hpp"
// #include "PhyTensorNURBSBasis.h"
// #include "PostProcess.h"
// #include "StiffnessAssembler.hpp"
// #include "Surface.hpp"
// #include "Utility.hpp"
// #include <boost/math/constants/constants.hpp>
// #include <boost/math/special_functions/legendre.hpp>

#include "BsplineBasis.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace Eigen;
using namespace std;


int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 4, 0, 1 );
    knot_vector.Insert( .3 );
    knot_vector.Insert( .3 );

    knot_vector.UniformRefine( 1 );
    knot_vector.printKnotVector();
    BsplineBasis<double> basis( knot_vector );
    basis.CompleteBezierDualInitialize();
    return 0;
}