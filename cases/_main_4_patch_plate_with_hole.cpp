
#include "BiharmonicInterfaceVisitor.hpp"
#include "BiharmonicStiffnessVisitor.hpp"
#include "BsplineBasis.h"
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
    const double nu = .3;
    const double E = 1e5;
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 2, 0, 1 );

    WeightedGeometryVector ctrlpts1{Vector3d( 0, 1, 1 ),
                                    Vector3d( 0, 1.75, 1 ),
                                    Vector3d( 0, 2.5, 1 ),
                                    Vector3d( 0.3535533905932737, 0.8535533905932737, 0.8535533905932737 ),
                                    Vector3d( 0.618718433538229, 1.493718433538229, 0.8535533905932737 ),
                                    Vector3d( 0.8838834764831843, 2.133883476483184, 0.8535533905932737 ),
                                    Vector3d( 0.6035533905932737, 0.6035533905932737, 0.8535533905932737 ),
                                    Vector3d( 1.056218433538229, 1.056218433538229, 0.8535533905932737 ),
                                    Vector3d( 1.508883476483184, 1.508883476483184, 0.8535533905932737 )};

    WeightedGeometryVector ctrlpts2{Vector3d( 0, 2.5, 1 ),
                                    Vector3d( 0, 3.25, 1 ),
                                    Vector3d( 0, 4, 1 ),
                                    Vector3d( 0.8838834764831843, 2.133883476483184, 0.8535533905932737 ),
                                    Vector3d( 1.14904851942814, 2.77404851942814, 0.8535533905932737 ),
                                    Vector3d( 1.414213562373095, 3.414213562373095, 0.8535533905932737 ),
                                    Vector3d( 1.508883476483184, 1.508883476483184, 0.8535533905932737 ),
                                    Vector3d( 1.96154851942814, 1.96154851942814, 0.8535533905932737 ),
                                    Vector3d( 2.414213562373095, 2.414213562373095, 0.8535533905932737 )};

    WeightedGeometryVector ctrlpts3{Vector3d( 0.6035533905932737, 0.6035533905932737, 0.8535533905932737 ),
                                    Vector3d( 1.056218433538229, 1.056218433538229, 0.8535533905932737 ),
                                    Vector3d( 1.508883476483184, 1.508883476483184, 0.8535533905932737 ),
                                    Vector3d( 0.8535533905932737, 0.3535533905932737, 0.8535533905932737 ),
                                    Vector3d( 1.493718433538229, 0.618718433538229, 0.8535533905932737 ),
                                    Vector3d( 2.133883476483184, 0.8838834764831843, 0.8535533905932737 ),
                                    Vector3d( 1, 0, 1 ),
                                    Vector3d( 1.75, 0, 1 ),
                                    Vector3d( 2.5, 0, 1 )};

    WeightedGeometryVector ctrlpts4{Vector3d( 1.508883476483184, 1.508883476483184, 0.8535533905932737 ),
                                    Vector3d( 1.96154851942814, 1.96154851942814, 0.8535533905932737 ),
                                    Vector3d( 2.414213562373095, 2.414213562373095, 0.8535533905932737 ),
                                    Vector3d( 2.133883476483184, 0.8838834764831843, 0.8535533905932737 ),
                                    Vector3d( 2.77404851942814, 1.14904851942814, 0.8535533905932737 ),
                                    Vector3d( 3.414213562373095, 1.414213562373095, 0.8535533905932737 ),
                                    Vector3d( 2.5, 0, 1 ),
                                    Vector3d( 3.25, 0, 1 ),
                                    Vector3d( 4, 0, 1 )};

    int degree, refine;
    cin >> degree >> refine;
    for ( int d = 2; d < degree; ++d )
    {
        for ( int r = 0; r < refine; ++r )
        {
            array<shared_ptr<PhyTensorNURBSBasis<2, 2, double>>, 4> domains;
            domains[0] = make_shared<PhyTensorNURBSBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, ctrlpts1 );
            domains[1] = make_shared<PhyTensorNURBSBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, ctrlpts2 );
            domains[2] = make_shared<PhyTensorNURBSBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, ctrlpts3 );
            domains[3] = make_shared<PhyTensorNURBSBasis<2, 2, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, ctrlpts4 );

            domains[0]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
            domains[0]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
            domains[1]->KnotsInsertion( 0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );
            domains[1]->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );

            domains[3]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
            domains[3]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
            domains[2]->KnotsInsertion( 0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );
            domains[2]->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );

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

            vector<int> boundary_indices_x;

            for ( int k = 0; k < 2; k++ )
            {
                auto i = cells[k];
                int id = i->GetID();
                int starting_dof = dof.StartingDof( id );

                auto local_boundary_indices = i->EdgePointerGetter( 3 )->Indices( 1, 0 );
                std::for_each( local_boundary_indices.begin(), local_boundary_indices.end(),
                               [&]( int& index ) { index += starting_dof; } );
                boundary_indices_x.insert( boundary_indices_x.end(), local_boundary_indices.begin(),
                                           local_boundary_indices.end() );
            }
            sort( boundary_indices_x.begin(), boundary_indices_x.end() );
            boundary_indices_x.erase( unique( boundary_indices_x.begin(), boundary_indices_x.end() ), boundary_indices_x.end() );

            vector<int> boundary_indices_y;

            for ( int k = 2; k < 4; k++ )
            {
                auto i = cells[k];
                int id = i->GetID();
                int starting_dof = dof.StartingDof( id );

                auto local_boundary_indices = i->EdgePointerGetter( 1 )->Indices( 1, 0 );
                std::for_each( local_boundary_indices.begin(), local_boundary_indices.end(),
                               [&]( int& index ) { index += starting_dof; } );
                boundary_indices_y.insert( boundary_indices_y.end(), local_boundary_indices.begin(),
                                           local_boundary_indices.end() );
            }
            sort( boundary_indices_y.begin(), boundary_indices_y.end() );
            boundary_indices_y.erase( unique( boundary_indices_y.begin(), boundary_indices_y.end() ), boundary_indices_y.end() );

            ConstraintAssembler<2, 2, double> constraint_assemble( dof );
            constraint_assemble.ConstraintCodimensionCreator( cells );
            constraint_assemble.Additional_Constraint( boundary_indices_x );
            SparseMatrix<double> sp1;
            constraint_assemble.AssembleByCodimension( sp1 );

            constraint_assemble.Additional_Constraint( boundary_indices_y );
            SparseMatrix<double> sp2;
            constraint_assemble.AssembleByCodimension( sp2 );
            SparseMatrix<double> identity1, identity2;
            identity1.resize( 2, 1 );
            identity2.resize( 2, 1 );
            identity1.coeffRef( 0, 0 ) = 1;
            identity2.coeffRef( 1, 0 ) = 1;
            sp1 = kroneckerProduct( sp1, identity1 ).eval();
            sp2 = kroneckerProduct( sp2, identity2 ).eval();
            SparseMatrix<double> sp;
            sp.resize( 2 * dof.TotalDof(), sp1.cols() + sp2.cols() );
            sp.leftCols( sp1.cols() ) = sp1;
            sp.rightCols( sp2.cols() ) = sp2;

            function<vector<double>( const VectorXd& )> stress_solution = []( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                double r = sqrt( x * x + y * y );
                double theta = acos( x / r );
                double T = 10;
                double sigma_rr, sigma_tt, sigma_rt;
                sigma_rr = T / 2.0 * ( 1 - pow( 1.0 / r, 2 ) ) +
                           T / 2.0 * ( 1 - 4 * pow( 1.0 / r, 2 ) + 3 * pow( 1.0 / r, 4 ) ) * cos( 2 * theta );
                sigma_tt = T / 2.0 * ( 1 + pow( 1.0 / r, 2 ) ) - T / 2.0 * ( 1 + 3 * pow( 1.0 / r, 4 ) ) * cos( 2 * theta );
                sigma_rt = -T / 2.0 * ( 1 + 2 * pow( 1.0 / r, 2 ) - 3 * pow( 1.0 / r, 4 ) ) * sin( 2 * theta );
                MatrixXd stress_tensor_polar( 2, 2 ), stress_tensor_cartisan( 2, 2 ), transform( 2, 2 );
                transform << cos( theta ), -sin( theta ), sin( theta ), cos( theta );
                stress_tensor_polar << sigma_rr, sigma_rt, sigma_rt, sigma_tt;
                stress_tensor_cartisan = transform * stress_tensor_polar * transform.transpose();
                return vector<double>{stress_tensor_cartisan( 0, 0 ), stress_tensor_cartisan( 1, 1 ),
                                      stress_tensor_cartisan( 0, 1 )};
            };

            function<vector<double>( const VectorXd& )> displacement_solution = [&nu, &E]( const VectorXd& u ) {
                double x = u( 0 );
                double y = u( 1 );
                double T = 10;
                double mu = E / 2 / ( 1 + nu );
                double a = 1;
                double kappa = 3 - 4 * nu;
                double r = sqrt( pow( x, 2 ) + pow( y, 2 ) );
                double theta = acos( x / r );
                Vector2d result;
                result( 0 ) = T * a / 8 / mu *
                              ( r / a * ( kappa + 1 ) * cos( theta ) +
                                2.0 * a / r * ( ( 1 + kappa ) * cos( theta ) + cos( 3 * theta ) ) -
                                2.0 * pow( a / r, 3 ) * cos( 3 * theta ) );
                result( 1 ) = T * a / 8 / mu *
                              ( r / a * ( kappa - 3 ) * sin( theta ) +
                                2.0 * a / r * ( ( 1 - kappa ) * sin( theta ) + sin( 3 * theta ) ) -
                                2.0 * pow( a / r, 3 ) * sin( 3 * theta ) );
                double sigma_rr, sigma_tt, sigma_rt;
                sigma_rr = T / 2.0 * ( 1 - pow( 1.0 / r, 2 ) ) +
                           T / 2.0 * ( 1 - 4 * pow( 1.0 / r, 2 ) + 3 * pow( 1.0 / r, 4 ) ) * cos( 2 * theta );
                sigma_tt = T / 2.0 * ( 1 + pow( 1.0 / r, 2 ) ) - T / 2.0 * ( 1 + 3 * pow( 1.0 / r, 4 ) ) * cos( 2 * theta );
                sigma_rt = -T / 2.0 * ( 1 + 2 * pow( 1.0 / r, 2 ) - 3 * pow( 1.0 / r, 4 ) ) * sin( 2 * theta );
                MatrixXd stress_tensor_polar( 2, 2 ), stress_tensor_cartisan( 2, 2 ), transform( 2, 2 );
                transform << cos( theta ), -sin( theta ), sin( theta ), cos( theta );
                stress_tensor_polar << sigma_rr, sigma_rt, sigma_rt, sigma_tt;
                stress_tensor_cartisan = transform * stress_tensor_polar * transform.transpose();
                return vector<double>{result( 0 ), result( 1 ), stress_tensor_cartisan( 0, 0 ),
                                      stress_tensor_cartisan( 1, 1 ), stress_tensor_cartisan( 0, 1 )};
            };

            StiffnessAssembler<Elasticity2DStiffnessVisitor<double>> stiffness_assemble( dof );
            SparseMatrix<double> stiffness_matrix;

            stiffness_matrix.resize( 2 * dof.TotalDof(), 2 * dof.TotalDof() );
            stiffness_assemble.Assemble( cells, stiffness_matrix );
            MatrixXd temp( 2 * dof.TotalDof(), 2 * dof.TotalDof() );
            NeumannBoundaryVisitor<double> neumann1( stress_solution );
            cells[3]->EdgePointerGetter( 2 )->Accept( neumann1 );
            NeumannBoundaryVisitor<double> neumann2( stress_solution );
            cells[1]->EdgePointerGetter( 2 )->Accept( neumann2 );
            SparseMatrix<double> rhs1, rhs2;
            neumann1.NeumannBoundaryAssembler( rhs1 );
            neumann2.NeumannBoundaryAssembler( rhs2 );
            VectorXd load_vector( 2 * dof.TotalDof() );

            load_vector.setZero();
            load_vector.segment( dof.StartingDof( cells[3]->GetID() ) * 2, domains[3]->GetDof() * 2 ) = rhs1;
            load_vector.segment( dof.StartingDof( cells[1]->GetID() ) * 2, domains[1]->GetDof() * 2 ) = rhs2;
            SparseMatrix<double> constrained_stiffness_matrix = sp.transpose() * stiffness_matrix * sp;
            VectorXd constrained_rhs = sp.transpose() * ( load_vector );
            ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.compute( constrained_stiffness_matrix );
            VectorXd Solution = sp * cg.solve( constrained_rhs );
            vector<KnotVector<double>> solutionDomain1, solutionDomain2, solutionDomain3, solutionDomain4;
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 0 ) );
            solutionDomain1.push_back( domains[0]->KnotVectorGetter( 1 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 0 ) );
            solutionDomain2.push_back( domains[1]->KnotVectorGetter( 1 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 0 ) );
            solutionDomain3.push_back( domains[2]->KnotVectorGetter( 1 ) );
            solutionDomain4.push_back( domains[3]->KnotVectorGetter( 0 ) );
            solutionDomain4.push_back( domains[3]->KnotVectorGetter( 1 ) );
            VectorXd controlDomain1 = Solution.segment( 2 * dof.StartingDof( cells[0]->GetID() ), 2 * domains[0]->GetDof() );
            VectorXd controlDomain2 = Solution.segment( 2 * dof.StartingDof( cells[1]->GetID() ), 2 * domains[1]->GetDof() );
            VectorXd controlDomain3 = Solution.segment( 2 * dof.StartingDof( cells[2]->GetID() ), 2 * domains[2]->GetDof() );
            VectorXd controlDomain4 = Solution.segment( 2 * dof.StartingDof( cells[3]->GetID() ), 2 * domains[3]->GetDof() );
            vector<shared_ptr<PhyTensorNURBSBasis<2, 2, double>>> solutions( 4 );
            solutions[0] = make_shared<PhyTensorNURBSBasis<2, 2, double>>( solutionDomain1, controlDomain1,
                                                                           domains[0]->WeightVectorGetter() );
            solutions[1] = make_shared<PhyTensorNURBSBasis<2, 2, double>>( solutionDomain2, controlDomain2,
                                                                           domains[1]->WeightVectorGetter() );
            solutions[2] = make_shared<PhyTensorNURBSBasis<2, 2, double>>( solutionDomain3, controlDomain3,
                                                                           domains[2]->WeightVectorGetter() );
            solutions[3] = make_shared<PhyTensorNURBSBasis<2, 2, double>>( solutionDomain4, controlDomain4,
                                                                           domains[3]->WeightVectorGetter() );
            PostProcess<double, 2> post_process( cells, solutions, displacement_solution );
            post_process.Plot( 200 );
        }
        cout << endl;
    }

    return 0;
}