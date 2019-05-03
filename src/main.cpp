
#include "BiharmonicInterfaceVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "BsplineBasis.h"
#include "CahnHilliardVisitor.hpp"
#include "ConstraintAssembler.hpp"
#include "DofMapper.hpp"
#include "Elasticity2DStiffnessVisitor.hpp"
#include "L2StiffnessVisitor.hpp"
#include "MembraneStiffnessVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include "PhyTensorNURBSBasis.h"
#include "PoissonStiffnessVisitor.hpp"
#include "PostProcess.h"
#include "StiffnessAssembler.hpp"
#include "Surface.hpp"
#include "Utility.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <time.h>
#include <unsupported/Eigen/KroneckerProduct>

// #define EIGEN_DONT_PARALLELIZE

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 3, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 3, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 3, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

using Vector2d = Matrix<double, 2, 1>;

const double Pi = 3.14159265358979323846264338327;

int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 1, 0, 1 );

    Vector3d point11( 0.0, 0.0, 0.0 ), point12( 0.0, 10.0, 0.0 ), point13( 2.0, 0.0, 0.0 ), point14( 2.0, 10.0, 0.0 );
    Vector3d point21( 0.0, 0.0, 0.0 ), point22( 0.0, 10.0, 0.0 ), point23( 0.0, 0.0, -2.0 ), point24( 0.0, 10.0, -2.0 );
    GeometryVector points1{point11, point12, point13, point14};
    GeometryVector points2{point21, point22, point23, point24};
    auto domain1 =
        make_shared<PhyTensorBsplineBasis<2, 3, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    auto domain2 =
        make_shared<PhyTensorBsplineBasis<2, 3, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );

    int degree, refine;
    cin >> degree >> refine;

    domain1->DegreeElevate( degree );
    domain2->DegreeElevate( degree );

    domain1->KnotsInsertion( 1, {1.0 / 4, 2.0 / 4, 3.0 / 4} );
    domain2->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );

    domain1->UniformRefine( refine );
    domain2->UniformRefine( refine );

    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain1 ) );
    cells[0]->SurfaceInitialize();
    cells.push_back( make_shared<Surface<3, double>>( domain2 ) );
    cells[1]->SurfaceInitialize();
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }
    for ( int i = 0; i < 1; i++ )
    {
        for ( int j = i + 1; j < 2; j++ )
        {
            cells[i]->Match( cells[j] );
        }
    }
    for ( auto i : cells )
    {
        i->PrintEdgeInfo();
    }
    KLShellConstraintAssembler<double> klconstraints( dof );
    klconstraints.ConstraintInitialize( cells );
    klconstraints.ConstraintCreator( cells );
    SparseMatrix<double> sp;
    klconstraints.AssembleConstraints( sp );
    return 0;
}