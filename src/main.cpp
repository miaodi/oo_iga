
#include "BiharmonicInterfaceVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "BsplineBasis.h"
#include "ConstraintAssembler.hpp"
#include "DofMapper.hpp"
#include "L2StiffnessVisitor.hpp"
#include "MembraneStiffnessVisitor.hpp"
#include "PhyTensorNURBSBasis.h"
#include "PoissonStiffnessVisitor.hpp"
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
    knot_vector.InitClosed( 2, 0, 1 );
    Vector2d point1( 0, 0 );
    Vector2d point2( 0, .3 );
    Vector2d point3( 0, .5 );
    Vector2d point4( .22, 0 );
    Vector2d point5( .25, .25 );
    Vector2d point6( .28, .5 );
    Vector2d point7( .5, 0 );
    Vector2d point8( .5, .23 );
    Vector2d point9( .5, .5 );
    Vector2d xMove( .5, 0 );
    Vector2d yMove( 0, .5 );

    GeometryVector points1( {point1, point2, point3, point4, point5, point6, point7, point8, point9} );
    GeometryVector points2( {point1 + xMove, point2 + xMove, point3 + xMove, point4 + xMove, point5 + xMove,
                             point6 + xMove, point7 + xMove, point8 + xMove, point9 + xMove} );
    GeometryVector points3( {point1 + yMove, point2 + yMove, point3 + yMove, point4 + yMove, point5 + yMove,
                             point6 + yMove, point7 + yMove, point8 + yMove, point9 + yMove} );
    GeometryVector points4( {point1 + xMove + yMove, point2 + xMove + yMove, point3 + xMove + yMove,
                             point4 + xMove + yMove, point5 + xMove + yMove, point6 + xMove + yMove,
                             point7 + xMove + yMove, point8 + xMove + yMove, point9 + xMove + yMove} );

    int degree, refine;
    cin >> degree >> refine;
    for ( int d = 0; d < degree; ++d )
    {
        for ( int r = 1; r < refine; ++r )
        {
            array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 4> domains;
            domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
            domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
            domains[2] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3 );
            domains[3] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4 );

            for ( auto& i : domains )
            {
                i->DegreeElevate( d );
            }
            for ( auto& i : domains )
            {
                i->UniformRefine( r );
            }
            vector<shared_ptr<Surface<2, double>>> cells;
            for ( int i = 0; i < 4; i++ )
            {
                cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
                cells[i]->SurfaceInitialize();
            }

            for ( int i = 0; i < 3; i++ )
            {
                for ( int j = i + 1; j < 4; j++ )
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
                        auto local_boundary_indices = i->EdgePointerGetter( j )->Indices( 1, 0 );
                        std::for_each( local_boundary_indices.begin(), local_boundary_indices.end(),
                                       [&]( int& index ) { index += starting_dof; } );
                        boundary_indices.insert( boundary_indices.end(), local_boundary_indices.begin(),
                                                 local_boundary_indices.end() );
                    }
                }
            }
            sort( boundary_indices.begin(), boundary_indices.end() );
            boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

            function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                return vector<double>{8 * pow( Pi, 2 ) * sin( 2 * Pi * x ) * sin( 2 * Pi * y )};
            };

            function<vector<double>( const VectorXd& )> analytical_solution = []( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                return vector<double>{sin( 2 * Pi * x ) * sin( 2 * Pi * y ), 2 * Pi * cos( 2 * Pi * x ) * sin( 2 * Pi * y ),
                                      2 * Pi * sin( 2 * Pi * x ) * cos( 2 * Pi * y )};
            };

            ConstraintAssembler<2, 2, double> constraint_assemble( dof );
            constraint_assemble.ConstraintCodimensionCreator( cells );
            constraint_assemble.Additional_Constraint( boundary_indices );
            SparseMatrix<double> sp1;
            constraint_assemble.AssembleByCodimension( sp1 );

            StiffnessAssembler<PoissonStiffnessVisitor<double>> stiffness_assemble( dof );
            SparseMatrix<double> stiffness_matrix, load_vector;

            stiffness_matrix.resize( dof.TotalDof(), dof.TotalDof() );
            load_vector.resize( dof.TotalDof(), 1 );
            stiffness_assemble.Assemble( cells, body_force, stiffness_matrix, load_vector );

            SparseMatrix<double> constrained_stiffness_matrix = sp1.transpose() * stiffness_matrix * sp1;
            SparseMatrix<double> constrained_rhs = sp1.transpose() * ( load_vector );
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute( constrained_stiffness_matrix );
            VectorXd Solution = sp1 * cg.solve( constrained_rhs );
            vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3, solutionDomain4;
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 0 ) );
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 1 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 0 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 1 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 0 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 1 ) );
            solutionDomain4.push_back( domains[3]->KnotVectorGetter( 0 ) );
            solutionDomain4.push_back( domains[3]->KnotVectorGetter( 1 ) );
            VectorXd controlDomain1 = Solution.segment( dof.StartingDof( cells[0]->GetID() ), domains[0]->GetDof() );
            VectorXd controlDomain2 = Solution.segment( dof.StartingDof( cells[1]->GetID() ), domains[1]->GetDof() );
            VectorXd controlDomain3 = Solution.segment( dof.StartingDof( cells[2]->GetID() ), domains[2]->GetDof() );
            VectorXd controlDomain4 = Solution.segment( dof.StartingDof( cells[3]->GetID() ), domains[3]->GetDof() );
            vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions( 4 );
            solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain1, controlDomain1 );
            solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain2, controlDomain2 );
            solutions[2] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain3, controlDomain3 );
            solutions[3] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain4, controlDomain4 );

            PostProcess<double> post_process( cells, solutions, analytical_solution );
            cout << "L2 error: " << post_process.RelativeL2Error() << " H1 error: " << post_process.RelativeH1Error() << endl;
            post_process.Plot(100);
        }
        cout << endl;
    }

    return 0;
}