
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
    knot_vector.InitClosed( 2, 0, 1 );
    Vector2d point1( 0, 0 );
    Vector2d point2( 0, .5 );
    Vector2d point3( 0, 1.0 );
    Vector2d point4( .5, 0 );
    Vector2d point5( .5, .5 );
    Vector2d point6( .5, 1.0 );
    Vector2d point7( 1.0, 0 );
    Vector2d point8( 1.0, .45 );
    Vector2d point9( 1.0, 1.0 );

    GeometryVector points1( {point1, point2, point3, point4, point5, point6, point7, point8, point9} );

    shared_ptr<PhyTensorBsplineBasis<2, 2, double>> domain =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    int ref;
    cin >> ref;
    SparseMatrix<double> constraint;

    {
        Vector2d xMove( 1, 0 );
        Vector2d yMove( 0, 1 );
        GeometryVector points1( {point1, point2, point3, point4, point5, point6, point7, point8, point9} );
        GeometryVector points2( {point1 + 0 * xMove + 1 * yMove, point2 + 0 * xMove + 1 * yMove, point3 + 0 * xMove + 1 * yMove,
                                 point4 + 0 * xMove + 1 * yMove, point5 + 0 * xMove + 1 * yMove, point6 + 0 * xMove + 1 * yMove,
                                 point7 + 0 * xMove + 1 * yMove, point8 + 0 * xMove + 1 * yMove, point9 + 0 * xMove + 1 * yMove} );

        array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 3> domains;
        domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
            std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
        domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(
            std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );
        for ( auto& i : domains )
        {
            i->DegreeElevate( 1 );
            i->UniformRefine( ref );
        }
        vector<shared_ptr<Surface<2, double>>> cells;
        for ( int i = 0; i < 2; i++ )
        {
            cells.push_back( make_shared<Surface<2, double>>( domains[i] ) );
            cells[i]->SurfaceInitialize();
        }

        for ( int i = 0; i < 1; i++ )
        {
            for ( int j = i + 1; j < 2; j++ )
            {
                cells[i]->Match( cells[j] );
            }
        }
        DofMapper dof;
        dof.Insert( cells[0]->GetID(), cells[0]->GetDomain()->GetDof() );

        ConstraintAssembler<2, 2, double> constraint_assemble( dof );
        constraint_assemble.ConstraintCreator( cells );
        constraint_assemble.AssembleByReducedKernel( constraint );
        constraint.prune( 1e-10, 1e-10 );
        // cout << MatrixXd( constraint ) << endl;
        // cout << constraint.rows() << " " << constraint.cols() << endl;
    }
    domain->DegreeElevate( 1 );
    domain->UniformRefine( ref );
    int dof = domain->GetDof();
    VectorXd c = VectorXd::Random( dof ) * .1 + VectorXd::Constant( dof, .63 );

    VectorXd ct = VectorXd::Zero( dof );

    double rho_inf = .5;
    double alpha_m = .5 * ( 3 - rho_inf ) / ( 1 + rho_inf );
    double alpha_f = 1.0 / ( 1 + rho_inf );
    double gamma = .5 + alpha_m - alpha_f;

    double t_final = 100.0;
    double t_current = .0;
    double dt = 1e-8;
    shared_ptr<Surface<2, double>> cell = make_shared<Surface<2, double>>( domain );

    cell->SurfaceInitialize();
    auto load = []( const VectorXd& u ) -> std::vector<double> { return std::vector<double>{0, 0}; };
    {
        auto target_function = []( const VectorXd& u ) -> std::vector<double> {
            double lower_bound = -.005;
            double upper_bound = .005;
            std::uniform_real_distribution<double> unif( lower_bound, upper_bound );
            std::default_random_engine re;
            double a_random_double = unif( re );
            return std::vector<double>{u( 0 ) + a_random_double};
        };
        L2StiffnessVisitor<double> l2( target_function );
        cell->Accept( l2 );
        const auto l2_stiffness_triplet = l2.GetStiffness();
        const auto l2_rhs_triplet = l2.GetRhs();
        SparseMatrix<double> l2_matrix, l2_load;
        l2_matrix.resize( dof, dof );
        l2_load.resize( dof, 1 );
        l2_matrix.setFromTriplets( l2_stiffness_triplet.begin(), l2_stiffness_triplet.end() );
        l2_load.setFromTriplets( l2_rhs_triplet.begin(), l2_rhs_triplet.end() );
        BiCGSTAB<SparseMatrix<double>> solver;
        solver.compute( constraint.transpose() * l2_matrix * constraint );
        VectorXd c_new = solver.solve( constraint.transpose() * l2_load );
        c = constraint * c_new;
    }

    int thd;
    cin >> thd;
    auto g_alpha = [&c, &ct, cell, &load, dof, &dt, &constraint, thd](
                       double alpha_m, double alpha_f, double gamma, VectorXd& c_next, VectorXd& ct_next, double steps ) -> bool {
        // predictor stage
        VectorXd c_pred = c;
        VectorXd ct_pred = ( gamma - 1 ) / gamma * ct;
        double relative_residual = 0;
        for ( int i = 0; i < steps; i++ )
        {
            // alpha-levels
            VectorXd c_alpha = c + alpha_f * ( c_pred - c );
            VectorXd ct_alpha = ct + alpha_m * ( ct_pred - ct );

            CH4thStiffnessVisitor<double> CH4thsv( load );
            CH2ndStiffnessVisitor<double> CH2ndsv( load );
            CHMassVisitor<double> CHmv( load );
            CH4thsv.ThreadSetter( thd );
            CH2ndsv.ThreadSetter( thd );
            CHmv.ThreadSetter( thd );
            CH4thsv.SetStateData( c_alpha.data() );
            CH2ndsv.SetStateData( c_alpha.data() );
            CHmv.SetStateData( ct_alpha.data() );

            cout << "start assemble CH4th stiffness" << endl;
            cell->Accept( CH4thsv );
            cout << "end assemble CH4th stiffness" << endl;
            cout << "start assemble CH2nd stiffness" << endl;
            cell->Accept( CH2ndsv );
            cout << "end assemble CH4th stiffness" << endl;
            cout << "start assemble CHmv stiffness" << endl;
            cell->Accept( CHmv );
            cout << "end assemble CHmv stiffness" << endl;

            const auto stiffness_triplet_ch4th = CH4thsv.GetStiffness();
            const auto load_triplet_ch4th = CH4thsv.GetRhs();
            SparseMatrix<double> stiffness_matrix_ch4th, load_vector_ch4th;
            load_vector_ch4th.resize( dof, 1 );
            stiffness_matrix_ch4th.resize( dof, dof );
            load_vector_ch4th.setFromTriplets( load_triplet_ch4th.begin(), load_triplet_ch4th.end() );
            stiffness_matrix_ch4th.setFromTriplets( stiffness_triplet_ch4th.begin(), stiffness_triplet_ch4th.end() );

            const auto stiffness_triplet_ch2nd = CH2ndsv.GetStiffness();
            const auto load_triplet_ch2nd = CH2ndsv.GetRhs();
            SparseMatrix<double> stiffness_matrix_ch2nd, load_vector_ch2nd;
            load_vector_ch2nd.resize( dof, 1 );
            stiffness_matrix_ch2nd.resize( dof, dof );
            load_vector_ch2nd.setFromTriplets( load_triplet_ch2nd.begin(), load_triplet_ch2nd.end() );
            stiffness_matrix_ch2nd.setFromTriplets( stiffness_triplet_ch2nd.begin(), stiffness_triplet_ch2nd.end() );

            const auto stiffness_triplet_chm = CHmv.GetStiffness();
            const auto load_triplet_chm = CHmv.GetRhs();
            SparseMatrix<double> stiffness_matrix_chm, load_vector_chm;
            load_vector_chm.resize( dof, 1 );
            stiffness_matrix_chm.resize( dof, dof );
            load_vector_chm.setFromTriplets( load_triplet_chm.begin(), load_triplet_chm.end() );
            stiffness_matrix_chm.setFromTriplets( stiffness_triplet_chm.begin(), stiffness_triplet_chm.end() );

            SparseMatrix<double> stiffness_matrx =
                constraint.transpose() *
                ( alpha_m * stiffness_matrix_chm + alpha_f * gamma * dt * ( stiffness_matrix_ch2nd + stiffness_matrix_ch4th ) ) *
                constraint;
            VectorXd load_vector = constraint.transpose() * ( -load_vector_chm - load_vector_ch2nd - load_vector_ch4th );

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

            VectorXd dct = constraint * solver.solve( load_vector );

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
        num_of_steps++;
        if ( num_of_steps % 20 == 0 || num_of_steps == 0 )
        {
            std::ofstream file;
            std::string name;
            name = "TIME_" + std::to_string( t_current ) + ".txt";
            file.open( name );
            for ( int x = 0; x <= 100; x++ )
            {
                for ( int y = 0; y <= 100; y++ )
                {
                    Vector2d u, uphy;
                    uphy << 1.0 * x / 100, 1.0 * y / 100;
                    domain->InversePts( uphy, u );
                    double val = 0;
                    auto eval = domain->EvalDerAllTensor( u );
                    for ( auto& i : *eval )
                    {
                        val += i.second[0] * c( i.first );
                    }
                    file << uphy( 0 ) << " " << uphy( 1 ) << " " << val << std::endl;
                }
            }
            file.close();
        }
    }

    return 0;
}