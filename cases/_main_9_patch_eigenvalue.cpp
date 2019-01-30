
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
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace Spectra;
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
    Vector2d point1( 0, 0 );
    Vector2d point2( 0, 1.0 / 3 );
    Vector2d point3( 1.0 / 3, 0 );
    Vector2d point4( 1.0 / 3, 1.0 / 3 );
    Vector2d xMove( 1.0 / 3, 0 );
    Vector2d yMove( 0, 1.0 / 3 );

    GeometryVector points1( {point1, point2, point3, point4} );
    GeometryVector points2( {point1 + 0 * xMove + 1 * yMove, point2 + 0 * xMove + 1 * yMove,
                             point3 + 0 * xMove + 1 * yMove, point4 + 0 * xMove + 1 * yMove} );
    GeometryVector points3( {point1 + 0 * xMove + 2 * yMove, point2 + 0 * xMove + 2 * yMove,
                             point3 + 0 * xMove + 2 * yMove, point4 + 0 * xMove + 2 * yMove} );

    GeometryVector points4( {point1 + 1 * xMove + 0 * yMove, point2 + 1 * xMove + 0 * yMove,
                             point3 + 1 * xMove + 0 * yMove, point4 + 1 * xMove + 0 * yMove} );
    GeometryVector points5( {point1 + 1 * xMove + 1 * yMove, point2 + 1 * xMove + 1 * yMove,
                             point3 + 1 * xMove + 1 * yMove, point4 + 1 * xMove + 1 * yMove} );
    GeometryVector points6( {point1 + 1 * xMove + 2 * yMove, point2 + 1 * xMove + 2 * yMove,
                             point3 + 1 * xMove + 2 * yMove, point4 + 1 * xMove + 2 * yMove} );

    GeometryVector points7( {point1 + 2 * xMove + 0 * yMove, point2 + 2 * xMove + 0 * yMove,
                             point3 + 2 * xMove + 0 * yMove, point4 + 2 * xMove + 0 * yMove} );
    GeometryVector points8( {point1 + 2 * xMove + 1 * yMove, point2 + 2 * xMove + 1 * yMove,
                             point3 + 2 * xMove + 1 * yMove, point4 + 2 * xMove + 1 * yMove} );
    GeometryVector points9( {point1 + 2 * xMove + 2 * yMove, point2 + 2 * xMove + 2 * yMove,
                             point3 + 2 * xMove + 2 * yMove, point4 + 2 * xMove + 2 * yMove} );

    int degree, refine;
    cin >> degree >> refine;
    array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 9> domains;
    domains[0] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    domains[1] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
    domains[2] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3 );
    domains[3] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4 );
    domains[4] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points5 );
    domains[5] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points6 );
    domains[6] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points7 );
    domains[7] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points8 );
    domains[8] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points9 );

    for ( auto& i : domains )
    {
        i->DegreeElevate( degree );
    }
    for ( auto& i : domains )
    {
        i->UniformRefine( refine );
    }

    domains[0]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    domains[0]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    domains[1]->KnotsInsertion( 0, {1.0 / 2} );
    domains[1]->KnotsInsertion( 1, {1.0 / 2} );
    domains[2]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    domains[2]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );

    domains[3]->KnotsInsertion( 0, {1.0 / 2} );
    domains[3]->KnotsInsertion( 1, {1.0 / 2} );
    domains[4]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    domains[4]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    domains[5]->KnotsInsertion( 0, {1.0 / 2} );
    domains[5]->KnotsInsertion( 1, {1.0 / 2} );

    domains[6]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    domains[6]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    domains[7]->KnotsInsertion( 0, {1.0 / 2} );
    domains[7]->KnotsInsertion( 1, {1.0 / 2} );
    domains[8]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    domains[8]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );

    // domains[3]->KnotsInsertion( 0, {1.0 / 3, 2.0 / 3} );
    // domains[3]->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    // domains[2]->KnotsInsertion( 0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );
    // domains[2]->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );

    vector<shared_ptr<Surface<2, double>>> cells;
    for ( int i = 0; i < 9; i++ )
    {
        cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
        cells[i]->SurfaceInitialize();
    }

    for ( int i = 0; i < 8; i++ )
    {
        for ( int j = i + 1; j < 9; j++ )
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
                boundary_indices.insert( boundary_indices.end(), local_boundary_indices.begin(), local_boundary_indices.end() );
            }
        }
    }
    sort( boundary_indices.begin(), boundary_indices.end() );
    boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

    ConstraintAssembler<2, 2, double> constraint_assemble( dof );
    constraint_assemble.ConstraintCreator( cells );
    constraint_assemble.Additional_Constraint( boundary_indices );
    SparseMatrix<double> sp1;
    constraint_assemble.AssembleByReducedKernel( sp1 );

    StiffnessAssembler<PoissonStiffnessVisitor<double>> stiffness_assemble( dof );
    StiffnessAssembler<L2StiffnessVisitor<double>> mass_assemble( dof );
    SparseMatrix<double> stiffness_matrix, mass_matrix;

    stiffness_matrix.resize( dof.TotalDof(), dof.TotalDof() );
    mass_matrix.resize( dof.TotalDof(), dof.TotalDof() );
    stiffness_assemble.Assemble( cells, stiffness_matrix );
    mass_assemble.Assemble( cells, mass_matrix );

    SparseMatrix<double> constrained_stiffness_matrix = sp1.transpose() * stiffness_matrix * sp1;
    SparseMatrix<double> constrained_mass_matrix = sp1.transpose() * mass_matrix * sp1;
    cout << "working on matrix\n";
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute( constrained_mass_matrix );
    MatrixXd Solution = solver.solve( constrained_stiffness_matrix );
    VectorXcd eigen = Solution.eigenvalues();
    std::sort( eigen.data(), eigen.data() + eigen.size(),
               []( std::complex<double> lhs, std::complex<double> rhs ) { return norm( rhs ) > norm( lhs ); } );
    VectorXd res = eigen.cwiseAbs().cwiseSqrt() / 3;
    cout << res.transpose() << endl;
    VectorXd exact( eigen.size() * eigen.size() );
    for ( int m = 0; m < eigen.size(); m++ )
    {
        for ( int n = 0; n < eigen.size(); n++ )
        {
            exact( eigen.size() * m + n ) = Pi * sqrt( pow( m + 1.0, 2 ) + pow( n + 1.0, 2 ) );
        }
    }
    std::sort( exact.data(), exact.data() + exact.size() );
    for ( int i = 0; i < eigen.size(); i++ )
    {
        res( i ) = res( i ) / exact( i );
    }
    std::ofstream file;
    file.open( "spectrum.txt" );
    for ( int i = 0; i < res.size(); i++ )
    {
        file << 1.0 * i / eigen.size() << " " << res( i ) << endl;
    }

    // // Construct matrix operation object using the wrapper class SparseGenMatProd
    // SparseGenMatProd<double> op( Solution );
    // cout << "working on eigenvalue\n";
    // // Construct eigen solver object, requesting the largest three eigenvalues
    // GenEigsSolver<double, LARGEST_MAGN, SparseGenMatProd<double>> eigs( &op, 2, 30 );

    // // Initialize and compute
    // eigs.init();
    // int nconv = eigs.compute();

    // // Retrieve results
    // Eigen::VectorXcd evalues;
    // if ( eigs.info() == SUCCESSFUL )
    //     evalues = eigs.eigenvalues();

    // std::cout << "Eigenvalues found:\n" << evalues.cwiseAbs().cwiseSqrt() << std::endl;

    return 0;
}