
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
    Vector2d point11( 0, 0 );
    Vector2d point12( 0, 6 );
    Vector2d point13( 0, 12 );
    Vector2d point14( 4.5, 4.5 );
    Vector2d point15( 2, 8 );
    Vector2d point16( 4, 12 );
    Vector2d point17( 6, 6 );
    Vector2d point18( 4, 9 );
    Vector2d point19( 6, 12 );

    Vector2d point21( 0, 0 );
    Vector2d point22( 1.5, 1.5 );
    Vector2d point23( 6, 6 );
    Vector2d point24( 7, 0 );
    Vector2d point25( 7, 2 );
    Vector2d point26( 9, 8 );
    Vector2d point27( 12, 0 );
    Vector2d point28( 12, 3 );
    Vector2d point29( 12, 6 );

    Vector2d point31( 6, 6 );
    Vector2d point32( 4, 9 );
    Vector2d point33( 6, 12 );
    Vector2d point34( 9, 8 );
    Vector2d point35( 10, 10 );
    Vector2d point36( 9, 12 );
    Vector2d point37( 12, 6 );
    Vector2d point38( 12, 9 );
    Vector2d point39( 12, 12 );
    GeometryVector points1( {point11, point12, point13, point14, point15, point16, point17, point18, point19} );
    GeometryVector points3( {point21, point22, point23, point24, point25, point26, point27, point28, point29} );
    GeometryVector points2( {point31, point32, point33, point34, point35, point36, point37, point38, point39} );

    int degree, refine;
    cin >> degree >> refine;
    for ( int d = 0; d < degree; ++d )
    {
        for ( int r = 0; r < refine; ++r )
        {
            array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 3> domains;
            domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
            domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
            domains[2] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3 );

            for ( auto& i : domains )
            {
                i->DegreeElevate( d );
            }

            domains[1]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
            domains[1]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
            domains[2]->KnotsInsertion( 0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );
            domains[2]->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );
            domains[0]->UniformRefine( 1 );
            for ( auto& i : domains )
            {
                i->UniformRefine( r );
            }
            vector<shared_ptr<Surface<2, double>>> cells;
            for ( int i = 0; i < 3; i++ )
            {
                cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
                cells[i]->SurfaceInitialize();
            }

            for ( int i = 0; i < 2; i++ )
            {
                for ( int j = i + 1; j < 3; j++ )
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

            double L = 12.0, t = .375, E = 4.8e5, nu = .38;
            double D = E * pow( t, 3 ) / 12 / ( 1 - pow( nu, 2 ) );
            function<vector<double>( const VectorXd& )> body_force = [E, t, L, D]( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                return vector<double>{sin( Pi * x / L ) * sin( Pi * y / L ) / D};
            };

            function<vector<double>( const VectorXd& )> analytical_solution = [E, t, nu, L, D]( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                return vector<double>{pow( L, 4 ) * sin( Pi * x / L ) * sin( Pi * y / L ) / D / pow( Pi, 4 ) / 4};
            };
            ConstraintAssembler<2, 2, double> constraint_assemble( dof );
            constraint_assemble.ConstraintCodimensionCreator( cells );
            constraint_assemble.Additional_Constraint( boundary_indices );
            SparseMatrix<double> sp1;
            constraint_assemble.AssembleByCodimension( sp1 );

            StiffnessAssembler<BiharmonicStiffnessVisitor<double>> stiffness_assemble( dof );
            SparseMatrix<double> stiffness_matrix, load_vector;

            stiffness_matrix.resize( dof.TotalDof(), dof.TotalDof() );
            load_vector.resize( dof.TotalDof(), 1 );
            stiffness_assemble.Assemble( cells, body_force, stiffness_matrix, load_vector );

            SparseMatrix<double> constrained_stiffness_matrix = sp1.transpose() * stiffness_matrix * sp1;
            SparseMatrix<double> constrained_rhs = sp1.transpose() * ( load_vector );
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute( constrained_stiffness_matrix );
            VectorXd Solution = sp1 * cg.solve( constrained_rhs );
            vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3;
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 0 ) );
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 1 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 0 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 1 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 0 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 1 ) );
            VectorXd controlDomain1 = Solution.segment( dof.StartingDof( cells[0]->GetID() ), domains[0]->GetDof() );
            VectorXd controlDomain2 = Solution.segment( dof.StartingDof( cells[1]->GetID() ), domains[1]->GetDof() );
            VectorXd controlDomain3 = Solution.segment( dof.StartingDof( cells[2]->GetID() ), domains[2]->GetDof() );
            vector<shared_ptr<PhyTensorBsplineBasis<2, 1, double>>> solutions( 3 );
            solutions[0] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain1, controlDomain1 );
            solutions[1] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain2, controlDomain2 );
            solutions[2] = make_shared<PhyTensorBsplineBasis<2, 1, double>>( solutionDomain3, controlDomain3 );

            PostProcess<double> post_process( cells, solutions, analytical_solution );
            cout << "L2 error: " << post_process.RelativeL2Error() << " Mx error: " << post_process.RelativeMxError( D, nu, L )
                 << " Mxy error: " << post_process.RelativeMxyError( D, nu, L ) << endl;
            post_process.Plot( 100 );
            // std::ofstream file;
            // file.open( "file.txt" );
            // for ( int i = 0; i < cells.size(); i++ )
            // {
            //     auto ctrpts = domains[i]->CtrPtsVecGetter();
            //     auto solpts = solutions[i]->CtrPtsVecGetter();

            //     file << "{";
            //     for ( int i = 0; i < 9; i++ )
            //     {
            //         file << "{";
            //         for ( int j = 0; j < 9; j++ )
            //         {
            //             file << "(" << ctrpts[9 * i + j]( 0 ) << ", " << ctrpts[9 * i + j]( 1 ) << ", "
            //                  << solpts[9 * i + j]( 0 ) << "),";
            //         }
            //         file << "},";
            //     }
            //     file << "}" << endl;
            // }
            // file.close();
        }
        cout << endl;
    }
    return 0;
}