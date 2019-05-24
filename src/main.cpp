
#include "BendingStiffnessVisitor.hpp"
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
    const double L = 10.35;
    const double R = 4.953;
    const double P = 40000;
    KnotVector<double> knot_vector_xi;
    knot_vector_xi.InitClosed( 1, 0, 1 );
    KnotVector<double> knot_vector_eta;
    knot_vector_eta.InitClosed( 3, 0, 1 );
    Vector3d v;
    v << 0, 0, 1;
    Vector3d point1( 0, 0, -R ), point2( 0, 2 * R, -R ), point3( 0, 2 * R, R ), point4( 0, 0, R );
    Vector3d point5( 0, 0, R ), point6( 0, -2 * R, R ), point7( 0, -2 * R, -R ), point8( 0, 0, -R );
    Vector3d x_translate( L / 2, 0, 0 );
    GeometryVector points1{point1,
                           point2,
                           point3,
                           point4,
                           point1 + x_translate,
                           point2 + x_translate,
                           point3 + x_translate,
                           point4 + x_translate};

    GeometryVector points2{point1 - x_translate,
                           point2 - x_translate,
                           point3 - x_translate,
                           point4 - x_translate,
                           point1,
                           point2,
                           point3,
                           point4};

    GeometryVector points3{point5,
                           point6,
                           point7,
                           point8,
                           point5 + x_translate,
                           point6 + x_translate,
                           point7 + x_translate,
                           point8 + x_translate};

    GeometryVector points4{point5 - x_translate,
                           point6 - x_translate,
                           point7 - x_translate,
                           point8 - x_translate,
                           point5,
                           point6,
                           point7,
                           point8};

    Vector1d weight1( 1 ), weight2( 1.0 / 3 );
    WeightVector weights{weight1, weight2, weight2, weight1, weight1, weight2, weight2, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector_xi, knot_vector_eta}, points1, weights );
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector_xi, knot_vector_eta}, points2, weights );
    auto domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector_xi, knot_vector_eta}, points3, weights );
    auto domain4 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector_xi, knot_vector_eta}, points4, weights );

    int degree, refine;
    cin >> degree >> refine;

    domain1->DegreeElevate( 1, degree );
    domain1->DegreeElevate( 0, degree + 2 );
    domain1->UniformRefine( refine );

    domain2->DegreeElevate( 1, degree );
    domain2->DegreeElevate( 0, degree + 2 );
    domain2->UniformRefine( refine );

    domain3->DegreeElevate( 1, degree );
    domain3->DegreeElevate( 0, degree + 2 );
    domain3->UniformRefine( refine );

    domain4->DegreeElevate( 1, degree );
    domain4->DegreeElevate( 0, degree + 2 );
    domain4->UniformRefine( refine );

    domain1->CreateCurrentConfig();
    domain2->CreateCurrentConfig();
    domain3->CreateCurrentConfig();
    domain4->CreateCurrentConfig();

    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain1 ) );
    cells.push_back( make_shared<Surface<3, double>>( domain2 ) );
    cells.push_back( make_shared<Surface<3, double>>( domain3 ) );
    cells.push_back( make_shared<Surface<3, double>>( domain4 ) );
    cells[0]->SurfaceInitialize();
    cells[1]->SurfaceInitialize();
    cells[2]->SurfaceInitialize();
    cells[3]->SurfaceInitialize();

    function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
        return vector<double>{0, 0, 0};
    };
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }

    cells[0]->EdgePointerGetter( 3 )->Match( cells[1]->EdgePointerGetter( 1 ) );
    cells[0]->EdgePointerGetter( 0 )->Match( cells[2]->EdgePointerGetter( 2 ) );
    cells[0]->EdgePointerGetter( 2 )->Match( cells[2]->EdgePointerGetter( 0 ) );

    cells[2]->EdgePointerGetter( 3 )->Match( cells[3]->EdgePointerGetter( 1 ) );
    cells[1]->EdgePointerGetter( 0 )->Match( cells[3]->EdgePointerGetter( 2 ) );
    cells[1]->EdgePointerGetter( 2 )->Match( cells[3]->EdgePointerGetter( 0 ) );

    VectorXd u( dof.TotalDof() );

    auto indices = cells[0]->VertexPointerGetter( 0 )->Indices( 3, 0 );
    vector<int> boundary_indices;
    auto start_index = dof.StartingDof( cells[0]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + i );
    }
    // indices = cells[1]->VertexPointerGetter( 1 )->Indices( 3, 0 );
    // start_index = dof.StartingDof( cells[1]->GetID() );
    // for ( auto& i : indices )
    // {
    //     boundary_indices.push_back( start_index + i );
    // }
    // indices = cells[2]->VertexPointerGetter( 3 )->Indices( 3, 0 );
    // start_index = dof.StartingDof( cells[2]->GetID() );
    // for ( auto& i : indices )
    // {
    //     boundary_indices.push_back( start_index + i );
    // }
    // indices = cells[3]->VertexPointerGetter( 2 )->Indices( 3, 0 );
    // start_index = dof.StartingDof( cells[3]->GetID() );
    // for ( auto& i : indices )
    // {
    //     boundary_indices.push_back( start_index + i );
    // }

    KLShellConstraintAssembler<double> ca( dof );
    ca.ConstraintInitialize( cells );
    ca.ConstraintCreator( cells );
    ca.Additional_Constraint( boundary_indices );
    SparseMatrix<double> constraint_basis;
    ca.AssembleConstraints( constraint_basis );

    indices = cells[0]->VertexPointerGetter( 3 )->Indices( 3, 0 );
    std::ofstream file;
    file.open( "data.txt" );
    int thd;
    cout << "How many threads?" << endl;
    cin >> thd;
    for ( int i = 1; i <= 20; i++ )
    {
        double err;
        double init_err = 0;

        do
        {
            SparseMatrix<double> stiffness_matrix_bend, stiffness_matrix_bend_1, stiffness_matrix_mem,
                stiffness_matrix_mem_1, F_int_bend, F_int_mem, F_ext_local;
            stiffness_matrix_bend.resize( dof.TotalDof(), dof.TotalDof() );
            stiffness_matrix_mem.resize( dof.TotalDof(), dof.TotalDof() );
            stiffness_matrix_mem_1.resize( dof.TotalDof(), dof.TotalDof() );
            stiffness_matrix_bend_1.resize( dof.TotalDof(), dof.TotalDof() );
            F_int_bend.resize( dof.TotalDof(), 1 );
            F_int_mem.resize( dof.TotalDof(), 1 );

            StiffnessAssembler<NonlinearBendingStiffnessVisitor<double, NonlinearBendingStiffnessType::Default>> bending_stiffness_assemble(
                dof );
            bending_stiffness_assemble.ThreadSetter( thd );
            bending_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_bend, F_int_bend );

            StiffnessAssembler<NonlinearMembraneStiffnessVisitor<double, NonlinearMembraneStiffnessType::Default>> membrane_stiffness_assemble(
                dof );
            membrane_stiffness_assemble.ThreadSetter( thd );
            membrane_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_mem, F_int_mem );
            StiffnessAssembler<NonlinearMembraneStiffnessVisitor<double, NonlinearMembraneStiffnessType::Fourth>> membrane_stiffness_assemble1(
                dof );
            membrane_stiffness_assemble1.ThreadSetter( thd );
            membrane_stiffness_assemble1.Assemble( cells, stiffness_matrix_mem_1 );
            StiffnessAssembler<NonlinearBendingStiffnessVisitor<double, NonlinearBendingStiffnessType::AllExcpet>> bending_stiffness_assemble1(
                dof );
            bending_stiffness_assemble1.ThreadSetter( thd );
            bending_stiffness_assemble1.Assemble( cells, stiffness_matrix_bend_1 );
            SparseMatrix<double> stiffness_matrix =
                stiffness_matrix_bend + stiffness_matrix_mem + stiffness_matrix_mem_1 + stiffness_matrix_bend_1;
            cout << "Constructed stiffness matrix.\n";

            VectorXd F_ext( dof.TotalDof() );

            F_ext.setZero();
            F_ext( indices[2] ) = P * .05 * i;
            VectorXd F = F_ext - F_int_mem - F_int_bend;
            cout << "Constructed load vector.\n";

            SparseMatrix<double> stiff_sol = ( constraint_basis.transpose() * stiffness_matrix * constraint_basis );
            VectorXd load_sol = ( constraint_basis.transpose() * F );

            SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
            // fill A and b;
            // Compute the ordering permutation vector from the structural pattern of A
            solver.analyzePattern( stiff_sol );
            // Compute the numerical factorization
            solver.factorize( stiff_sol );
            // Use the factors to solve the linear system
            u = constraint_basis * solver.solve( load_sol );
            cout << "Solved the problem.\n";

            Map<MatrixXd> umat( u.data(), 3, u.size() / 3 );
            err = load_sol.norm();

            if ( init_err == 0 )
                init_err = err;
            err /= init_err;
            cout << err << endl;
            domain1->UpdateCurrentGeometryVector( umat.middleCols( 0, domain1->GetDof() ) );
            domain2->UpdateCurrentGeometryVector( umat.middleCols( domain1->GetDof(), domain2->GetDof() ) );
            domain3->UpdateCurrentGeometryVector( umat.middleCols( domain1->GetDof() + domain2->GetDof(), domain3->GetDof() ) );
            domain4->UpdateCurrentGeometryVector(
                umat.middleCols( domain1->GetDof() + domain2->GetDof() + domain3->GetDof(), domain4->GetDof() ) );

        } while ( err > 1e-7 );
        Vector2d pos;
        pos << 0, 1;
        VectorXd res = ( domain1->CurrentConfigGetter() ).AffineMap( pos ) - domain1->AffineMap( pos );
        file << i << " " << res( 2 ) / 2 << endl;
    }
    file.close();

    // array<ofstream, 3> myfiles;

    // myfiles[0].open( "domain1.txt" );
    // myfiles[1].open( "domain2.txt" );
    // myfiles[2].open( "domain3.txt" );
    // Vector2d pos;
    // for ( int i = 0; i < 51; i++ )
    // {
    //     for ( int j = 0; j < 51; j++ )
    //     {
    //         pos << 1.0 * i / 50, 1.0 * j / 50;
    //         VectorXd res = ( domain1->CurrentConfigGetter() ).AffineMap( pos );
    //         myfiles[0] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    //         res = ( domain2->CurrentConfigGetter() ).AffineMap( pos );
    //         myfiles[1] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    //         res = ( domain3->CurrentConfigGetter() ).AffineMap( pos );
    //         myfiles[2] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;

    //         // VectorXd res = domain1->AffineMap( pos );
    //         // myfiles[0] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    //         // res = domain2->AffineMap( pos );
    //         // myfiles[1] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    //         // res = domain3->AffineMap( pos );
    //         // myfiles[2] << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    //     }
    // }
    return 0;
}