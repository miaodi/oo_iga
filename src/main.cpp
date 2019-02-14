
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
    Vector2d point2( 0, .5 );
    Vector2d point3( .5, 0 );
    Vector2d point4( .5, .5 );
    Vector2d xMove( .5, 0 );
    Vector2d yMove( 0, .5 );

    GeometryVector points1( {point1, point2, point3, point4} );
    GeometryVector points2( {point1 + 0 * xMove + 1 * yMove, point2 + 0 * xMove + 1 * yMove,
                             point3 + 0 * xMove + 1 * yMove, point4 + 0 * xMove + 1 * yMove} );
    GeometryVector points3( {point1 + 1 * xMove + 0 * yMove, point2 + 1 * xMove + 0 * yMove,
                             point3 + 1 * xMove + 0 * yMove, point4 + 1 * xMove + 0 * yMove} );
    GeometryVector points4( {point1 + 1 * xMove + 1 * yMove, point2 + 1 * xMove + 1 * yMove,
                             point3 + 1 * xMove + 1 * yMove, point4 + 1 * xMove + 1 * yMove} );

    array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 4> domains;
    domains[0] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    domains[1] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
    domains[2] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3 );
    domains[3] =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4 );
    for ( auto& i : domains )
    {
        i->DegreeElevate( 2 );
    }

    domains[0]->UniformRefineDof( 0, 32 );
    domains[0]->UniformRefineDof( 1, 32 );

    domains[1]->UniformRefineDof( 0, 36 );
    domains[1]->UniformRefineDof( 1, 36 );

    domains[2]->UniformRefineDof( 0, 36 );
    domains[2]->UniformRefineDof( 1, 36 );

    domains[3]->UniformRefineDof( 0, 32 );
    domains[3]->UniformRefineDof( 1, 32 );

    vector<shared_ptr<Surface<2, double>>> cells;
    for ( int i = 0; i < 4; i++ )
    {
        cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
        cells[i]->SurfaceInitialize();
    }
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), i->GetDomain()->GetDof() );
    }
    for ( int i = 0; i < 3; i++ )
    {
        for ( int j = i + 1; j < 4; j++ )
        {
            cells[i]->Match( cells[j] );
        }
    }
    SparseMatrix<double, RowMajor> constraint;
    ConstraintAssembler<2, 2, double> constraint_assemble( dof );
    constraint_assemble.ConstraintCreator( cells );
    constraint_assemble.AssembleConstraint( constraint );

    for ( auto i : cells )
    {
        for ( int j = 0; j < 4; j++ )
        {
            auto edge = i->EdgePointerGetter( j );
            if ( !edge->IsMatched() )
            {
                if ( edge->GetOrient() == Orientation::east )
                {
                    auto c0_ind = i->GetDomain()->HyperPlaneIndices( 0, 0 );
                    auto c1_ind = i->GetDomain()->HyperPlaneIndices( 0, 1 );
                    for ( int num = 0; num < c0_ind->size(); num++ )
                    {
                        constraint.conservativeResize( constraint.rows() + 1, dof.TotalDof() );
                        constraint.coeffRef( constraint.rows() - 1, ( *c0_ind )[num] + dof.StartingDof( i->GetID() ) ) = 1;
                        constraint.coeffRef( constraint.rows() - 1, ( *c1_ind )[num] + dof.StartingDof( i->GetID() ) ) = -1;
                    }
                }
                else if ( edge->GetOrient() == Orientation::west )
                {
                    auto c0_ind = i->GetDomain()->HyperPlaneIndices( 0, i->GetDomain()->GetDof( 0 ) - 1 );
                    auto c1_ind = i->GetDomain()->HyperPlaneIndices( 0, i->GetDomain()->GetDof( 0 ) - 2 );
                    for ( int num = 0; num < c0_ind->size(); num++ )
                    {
                        constraint.conservativeResize( constraint.rows() + 1, dof.TotalDof() );
                        constraint.coeffRef( constraint.rows() - 1, ( *c0_ind )[num] + dof.StartingDof( i->GetID() ) ) = 1;
                        constraint.coeffRef( constraint.rows() - 1, ( *c1_ind )[num] + dof.StartingDof( i->GetID() ) ) = -1;
                    }
                }
                else if ( edge->GetOrient() == Orientation::south )
                {
                    auto c0_ind = i->GetDomain()->HyperPlaneIndices( 1, 0 );
                    auto c1_ind = i->GetDomain()->HyperPlaneIndices( 1, 1 );
                    for ( int num = 0; num < c0_ind->size(); num++ )
                    {
                        constraint.conservativeResize( constraint.rows() + 1, dof.TotalDof() );
                        constraint.coeffRef( constraint.rows() - 1, ( *c0_ind )[num] + dof.StartingDof( i->GetID() ) ) = 1;
                        constraint.coeffRef( constraint.rows() - 1, ( *c1_ind )[num] + dof.StartingDof( i->GetID() ) ) = -1;
                    }
                }
                else
                {
                    auto c0_ind = i->GetDomain()->HyperPlaneIndices( 1, i->GetDomain()->GetDof( 0 ) - 1 );
                    auto c1_ind = i->GetDomain()->HyperPlaneIndices( 1, i->GetDomain()->GetDof( 0 ) - 2 );
                    for ( int num = 0; num < c0_ind->size(); num++ )
                    {
                        constraint.conservativeResize( constraint.rows() + 1, dof.TotalDof() );
                        constraint.coeffRef( constraint.rows() - 1, ( *c0_ind )[num] + dof.StartingDof( i->GetID() ) ) = 1;
                        constraint.coeffRef( constraint.rows() - 1, ( *c1_ind )[num] + dof.StartingDof( i->GetID() ) ) = -1;
                    }
                }
            }
        }
    }
    MatrixXd dense_constraint = constraint;
    FullPivLU<MatrixXd> lu_decomp( dense_constraint );
    MatrixXd basis = lu_decomp.kernel();
    VectorXd c;
    VectorXd ct = VectorXd::Zero( dof.TotalDof() );

    double rho_inf = .5;
    double alpha_m = .5 * ( 3 - rho_inf ) / ( 1 + rho_inf );
    double alpha_f = 1.0 / ( 1 + rho_inf );
    double gamma = .5 + alpha_m - alpha_f;

    double t_final = 100.0;
    double t_current = .0;
    double dt = 1e-8;

    auto load = []( const VectorXd& u ) -> std::vector<double> { return std::vector<double>{0, 0}; };
    {
        auto target_function = []( const VectorXd& u ) -> std::vector<double> {
            // Type of random number distribution
            std::uniform_real_distribution<double> dist( -.005, .005 ); //(min, max)

            // Mersenne Twister: Good quality random number generator
            std::mt19937 rng;
            // Initialize with non-deterministic seeds
            rng.seed( std::random_device{}() );
            return std::vector<double>{.8 * ( u( 0 ) - .5 ) + .5 + dist( rng )};
        };

        SparseMatrix<double> l2_matrix, l2_load;
        l2_matrix.resize( dof.TotalDof(), dof.TotalDof() );
        l2_load.resize( dof.TotalDof(), 1 );

        StiffnessAssembler<L2StiffnessVisitor<double>> stiffness_assemble( dof );
        stiffness_assemble.Assemble( cells, target_function, l2_matrix, l2_load );
        BiCGSTAB<SparseMatrix<double>> solver;
        SparseMatrix<double> stiffness_sol = ( basis.transpose() * l2_matrix * basis ).sparseView();
        SparseMatrix<double> rhs_sol = ( basis.transpose() * l2_load ).sparseView();
        solver.compute( stiffness_sol );
        VectorXd c_new = solver.solve( rhs_sol );
        c = basis * c_new;
    }


    int thd;
    cout << "How many threads?" << endl;
    cin >> thd;
    auto g_alpha = [&c, &ct, &cells, &load, &dof, &dt, &basis, thd]( double alpha_m, double alpha_f, double gamma,
                                                                     VectorXd& c_next, VectorXd& ct_next, double steps ) -> bool {
        // predictor stage
        VectorXd c_pred = c;
        VectorXd ct_pred = ( gamma - 1 ) / gamma * ct;
        double relative_residual = 0;
        for ( int i = 0; i < steps; i++ )
        {
            // alpha-levels
            VectorXd c_alpha = c + alpha_f * ( c_pred - c );
            VectorXd ct_alpha = ct + alpha_m * ( ct_pred - ct );

            StiffnessAssembler<CH4thStiffnessVisitor<double>> CH4thsv( dof );
            StiffnessAssembler<CH2ndStiffnessVisitor<double>> CH2ndsv( dof );
            StiffnessAssembler<CHMassVisitor<double>> CHmv( dof );
            CH4thsv.ThreadSetter( thd );
            CH2ndsv.ThreadSetter( thd );
            CHmv.ThreadSetter( thd );
            CH4thsv.SetStateDatas( c_alpha.data(), ct_alpha.data() );
            CH2ndsv.SetStateDatas( c_alpha.data(), ct_alpha.data() );
            CHmv.SetStateDatas( c_alpha.data(), ct_alpha.data() );

            cout << "start assemble CH4th stiffness" << endl;
            SparseMatrix<double> stiffness_matrix_ch4th, load_vector_ch4th;
            load_vector_ch4th.resize( dof.TotalDof(), 1 );
            stiffness_matrix_ch4th.resize( dof.TotalDof(), dof.TotalDof() );
            CH4thsv.Assemble( cells, load, stiffness_matrix_ch4th, load_vector_ch4th );
            cout << "end assemble CH4th stiffness" << endl;

            cout << "start assemble CH2nd stiffness" << endl;
            SparseMatrix<double> stiffness_matrix_ch2nd, load_vector_ch2nd;
            load_vector_ch2nd.resize( dof.TotalDof(), 1 );
            stiffness_matrix_ch2nd.resize( dof.TotalDof(), dof.TotalDof() );
            CH2ndsv.Assemble( cells, load, stiffness_matrix_ch2nd, load_vector_ch2nd );
            cout << "end assemble CH4th stiffness" << endl;

            cout << "start assemble CHmv stiffness" << endl;
            SparseMatrix<double> stiffness_matrix_chm, load_vector_chm;
            load_vector_chm.resize( dof.TotalDof(), 1 );
            stiffness_matrix_chm.resize( dof.TotalDof(), dof.TotalDof() );
            CHmv.Assemble( cells, load, stiffness_matrix_chm, load_vector_chm );
            cout << "end assemble CHmv stiffness" << endl;

            SparseMatrix<double> stiffness_matrx =
                ( basis.transpose() *
                  ( alpha_m * stiffness_matrix_chm + alpha_f * gamma * dt * ( stiffness_matrix_ch2nd + stiffness_matrix_ch4th ) ) * basis )
                    .sparseView();
            VectorXd load_vector = basis.transpose() * ( -load_vector_chm - load_vector_ch2nd - load_vector_ch4th );

            if ( i == 0 )
                relative_residual = load_vector.template lpNorm<Infinity>();
            double err = load_vector.template lpNorm<Infinity>() / relative_residual;
            cout << "Norm of residual: " << err << endl;
            if ( err < 1e-4 )
            {
                c_next = c_pred;
                ct_next = ct_pred;
                return true;
            }

            BiCGSTAB<SparseMatrix<double>> solver;
            solver.compute( stiffness_matrx );
            solver.setTolerance( 1e-16 );

            VectorXd dct = basis * solver.solve( load_vector );

            ct_pred.noalias() += dct;
            c_pred.noalias() += gamma * dt * dct;
        }
        return false;
    };

    int num_of_steps = 0;
    while ( t_current < t_final )
    {
        cout << "Time step: " << num_of_steps << endl;
        cout << " current time: " << t_current << ", current time step size: " << dt << endl;
        VectorXd c_next_alpha, ct_next_alpha, c_next_be, ct_next_be;

        if ( !g_alpha( alpha_m, alpha_f, gamma, c_next_alpha, ct_next_alpha, 30 ) )
        {
            cout << " generalized alpha does not converge\n";
            return 1;
        }
        if ( !g_alpha( 1, 1, 1, c_next_be, ct_next_be, 60 ) )
        {
            cout << " backward-euler does not converge\n";
            return 2;
        }
        double err = ( c_next_alpha - c_next_be ).template lpNorm<Infinity>() / c_next_alpha.template lpNorm<Infinity>();
        cout << "difference between generalized alpha and backward euler: " << err << endl;
        if ( err > 1e-3 )
        {
            dt *= sqrt( .85 * 1e-3 / err );
        }
        else
        {
            t_current += dt;
            c = c_next_alpha;
            ct = ct_next_alpha;
            dt *= sqrt( .85 * 1e-3 / err );
        }

        if ( num_of_steps % 20 == 0 || num_of_steps == 0 )
        {
            std::ofstream file;
            std::string name;
            name = "TIME_" + std::to_string( t_current ) + ".txt";
            file.open( name );
            for ( int x = 0; x <= 50; x++ )
            {
                for ( int y = 0; y <= 50; y++ )
                {
                    Vector2d u, uphy;
                    uphy << 1.0 * x / 50, 1.0 * y / 50;
                    // domain->InversePts( uphy, u );
                    double val = 0;
                    auto eval = domains[0]->EvalDerAllTensor( uphy );
                    for ( auto& i : *eval )
                    {
                        val += i.second[0] * c( i.first + dof.StartingDof( cells[0]->GetID() ) );
                    }
                    u = domains[0]->AffineMap( uphy );
                    file << u( 0 ) << " " << u( 1 ) << " " << val << std::endl;
                }
                for ( int y = 0; y <= 50; y++ )
                {
                    Vector2d u, uphy;
                    uphy << 1.0 * x / 50, 1.0 * y / 50;
                    // domain->InversePts( uphy, u );
                    double val = 0;
                    auto eval = domains[1]->EvalDerAllTensor( uphy );
                    for ( auto& i : *eval )
                    {
                        val += i.second[0] * c( i.first + dof.StartingDof( cells[1]->GetID() ) );
                    }
                    u = domains[1]->AffineMap( uphy );
                    file << u( 0 ) << " " << u( 1 ) << " " << val << std::endl;
                }
            }
            for ( int x = 0; x <= 50; x++ )
            {
                for ( int y = 0; y <= 50; y++ )
                {
                    Vector2d u, uphy;
                    uphy << 1.0 * x / 50, 1.0 * y / 50;
                    // domain->InversePts( uphy, u );
                    double val = 0;
                    auto eval = domains[2]->EvalDerAllTensor( uphy );
                    for ( auto& i : *eval )
                    {
                        val += i.second[0] * c( i.first + dof.StartingDof( cells[2]->GetID() ) );
                    }
                    u = domains[2]->AffineMap( uphy );
                    file << u( 0 ) << " " << u( 1 ) << " " << val << std::endl;
                }
                for ( int y = 0; y <= 50; y++ )
                {
                    Vector2d u, uphy;
                    uphy << 1.0 * x / 50, 1.0 * y / 50;
                    // domain->InversePts( uphy, u );
                    double val = 0;
                    auto eval = domains[3]->EvalDerAllTensor( uphy );
                    for ( auto& i : *eval )
                    {
                        val += i.second[0] * c( i.first + dof.StartingDof( cells[3]->GetID() ) );
                    }
                    u = domains[3]->AffineMap( uphy );
                    file << u( 0 ) << " " << u( 1 ) << " " << val << std::endl;
                }
            }
            file.close();
        }
        num_of_steps++;
    }

    return 0;
}