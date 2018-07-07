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
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 2, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

const double Pi = 3.14159265358979323846264338327;

int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 1, 0, 1 );
    Vector2d point1( 0, 4 );
    Vector2d point2( 0, 0 );
    Vector2d point3( 1, 3 );
    Vector2d point4( 1, 1 );
    Vector2d point5( 3, 3 );
    Vector2d point6( 3, 1 );
    Vector2d point7( 4, 4 );
    Vector2d point8( 4, 0 );

    GeometryVector points1( {point1, point3, point2, point4} );
    GeometryVector points2( {point2, point4, point8, point6} );
    GeometryVector points3( {point4, point3, point6, point5} );
    GeometryVector points4( {point6, point5, point8, point7} );
    GeometryVector points5( {point3, point1, point5, point7} );

    array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 5> domains;
    domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
    domains[2] = make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3 );
    domains[3] = make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4 );
    domains[4] = make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points5 );

    for ( auto& i : domains )
    {
        i->DegreeElevate( 2 );
    }

    for ( auto& i : domains )
    {
        i->UniformRefine( 5 );
    }
    vector<shared_ptr<Surface<2, double>>> cells;
    for ( int i = 0; i < 5; i++ )
    {
        cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
        cells[i]->SurfaceInitialize();
    }

    for ( int i = 0; i < 4; i++ )
    {
        for ( int j = i + 1; j < 5; j++ )
        {
            cells[i]->Match( cells[j] );
        }
    }
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), i->GetDomain()->GetDof() );
    }

    vector<int> boundary_indices;

    for ( auto& i : cells )
    {
        int id = i->GetID();
        int starting_dof = dof.StartingDof( id );
        for ( int j = 0; j < 4; j++ )
        {
            if ( !i->EdgePointerGetter( j )->IsMatched() )
            {
                auto local_boundary_indices = i->EdgePointerGetter( j )->Indices( 1, 1 );
                std::for_each( local_boundary_indices.begin(), local_boundary_indices.end(), [&]( int& index ) { index += starting_dof; } );
                boundary_indices.insert( boundary_indices.end(), local_boundary_indices.begin(), local_boundary_indices.end() );
            }
        }
    }
    sort( boundary_indices.begin(), boundary_indices.end() );
    boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

    function<vector<double>( const VectorXd& )> analytical_solution = [&]( const VectorXd& u ) {
        double x = u( 0 );
        double y = u( 1 );
        return vector<double>{sin( Pi * x ) * sin( Pi * y ), Pi * cos( Pi * x ) * sin( Pi * y ), Pi * sin( Pi * x ) * cos( Pi * y )};
    };

    function<vector<double>( const VectorXd& )> body_force = [&]( const VectorXd& u ) {
        return vector<double>{4 * pow( Pi, 4 ) * sin( Pi * u( 0 ) ) * sin( Pi * u( 1 ) )};
    };
    BoundaryAssembler<Surface<2, double>> boundary_visitor( dof );
    boundary_visitor.BoundaryValueCreator( cells, analytical_solution );

    return 0;
}