#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "MembraneStiffnessVisitor.hpp"
#include "BendingStiffnessVisitor.hpp"
#include "PostProcess.hpp"
#include "H1DomainSemiNormVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include "DofMapper.hpp"
#include <fstream>
#include <time.h>
#include <boost/math/constants/constants.hpp>
#include "BiharmonicInterfaceVisitor.hpp"
#include "StiffnessAssembler.hpp"
#include "ConstraintAssembler.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 3, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 3, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 3, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(4, 0, 1);
    knot_vector.UniformRefine(3);
    knot_vector.printKnotVector();
    BsplineBasis<double> a(knot_vector);
    a.ModifyBoundaryInitialize();
    return 0;
}