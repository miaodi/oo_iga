
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
    const double L = 10;
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 2, 0, 1 );
    Vector3d v;
    v << 0, 0, 1;
    Vector3d inner( 0, 6, 0 ), outer( 0, 10, 0 ), middle( 0, 8, 0 );
    Vector3d point1 = Accessory::RotationMatrix( v, sin( -2 * M_PI / 3 ), cos( -2 * M_PI / 3 ) ) * outer;
    Vector3d point2 = Accessory::RotationMatrix( v, sin( -2 * M_PI / 3 ), cos( -2 * M_PI / 3 ) ) * middle;
    Vector3d point3 = Accessory::RotationMatrix( v, sin( -2 * M_PI / 3 ), cos( -2 * M_PI / 3 ) ) * inner;

    Vector3d point4 = Accessory::RotationMatrix( v, sin( -M_PI / 3 ), cos( -M_PI / 3 ) ) * outer * 2;
    Vector3d point5 = Accessory::RotationMatrix( v, sin( -M_PI / 3 ), cos( -M_PI / 3 ) ) * middle * 2;
    Vector3d point6 = Accessory::RotationMatrix( v, sin( -M_PI / 3 ), cos( -M_PI / 3 ) ) * inner * 2;

    Vector3d point7 = outer;
    Vector3d point8 = middle;
    Vector3d point9 = inner;

    GeometryVector points1{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    GeometryVector points2{Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point1,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point2,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point3,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point4,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point5,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point6,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point7,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point8,
                           Accessory::RotationMatrix( v, sin( 2 * M_PI / 3 ), cos( 2 * M_PI / 3 ) ) * point9};

    GeometryVector points3{Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point1,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point2,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point3,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point4,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point5,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point6,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point7,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point8,
                           Accessory::RotationMatrix( v, sin( 4 * M_PI / 3 ), cos( 4 * M_PI / 3 ) ) * point9};
    Vector1d weight1( 1 ), weight2( 1.0 / 2 ), weight3( 1 );
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1, weights );
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2, weights );
    auto domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3, weights );

    int degree, refine;
    cin >> degree >> refine;

    domain1->DegreeElevate( degree );
    // domain1->KnotsInsertion( 0, {.2, .4, .6, .8} );
    // domain1->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    domain1->UniformRefine( refine );

    domain2->DegreeElevate( degree );
    // domain2->KnotsInsertion( 0, {.2, .4, .6, .8} );
    // domain2->KnotsInsertion( 1, {.2, .4, .6, .8} );
    domain2->UniformRefine( refine );

    domain3->DegreeElevate( degree );
    // domain3->KnotsInsertion( 0, {.2, .4, .6, .8} );
    // domain3->KnotsInsertion( 1, {1.0 / 3, 2.0 / 3} );
    domain3->UniformRefine( refine );

    domain1->CreateCurrentConfig();
    domain2->CreateCurrentConfig();
    domain3->CreateCurrentConfig();

    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain1 ) );
    cells.push_back( make_shared<Surface<3, double>>( domain2 ) );
    cells.push_back( make_shared<Surface<3, double>>( domain3 ) );
    cells[0]->SurfaceInitialize();
    cells[1]->SurfaceInitialize();
    cells[2]->SurfaceInitialize();

    function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
        return vector<double>{0, 0, 0};
    };
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }
    cells[0]->Match( cells[1] );
    cells[1]->Match( cells[2] );

    VectorXd u( dof.TotalDof() );

    auto indices = cells[0]->EdgePointerGetter( 3 )->Indices( 3, 1 );
    vector<int> boundary_indices;
    auto start_index = dof.StartingDof( cells[0]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + i );
    }

    KLShellConstraintAssembler<double> ca( dof );
    ca.ConstraintInitialize( cells );

    std::ofstream file;
    file.open( "data.txt" );

    for ( int i = 1; i <= 10; i++ )
    {
        double err;
        double init_err = 0;

        ca.ConstraintCreator( cells );
        ca.Additional_Constraint( boundary_indices );
        SparseMatrix<double> constraint_basis;
        ca.AssembleConstraints( constraint_basis );
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
            bending_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_bend, F_int_bend );

            StiffnessAssembler<NonlinearMembraneStiffnessVisitor<double, NonlinearMembraneStiffnessType::Default>> membrane_stiffness_assemble(
                dof );
            membrane_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_mem, F_int_mem );
            StiffnessAssembler<NonlinearMembraneStiffnessVisitor<double, NonlinearMembraneStiffnessType::Fourth>> membrane_stiffness_assemble1(
                dof );
            membrane_stiffness_assemble1.Assemble( cells, stiffness_matrix_mem_1 );
            StiffnessAssembler<NonlinearBendingStiffnessVisitor<double, NonlinearBendingStiffnessType::AllExcpet>> bending_stiffness_assemble1(
                dof );
            bending_stiffness_assemble1.Assemble( cells, stiffness_matrix_bend_1 );
            SparseMatrix<double> stiffness_matrix =
                stiffness_matrix_bend + stiffness_matrix_mem + stiffness_matrix_mem_1 + stiffness_matrix_bend_1;
            cout << "Constructed stiffness matrix.\n";

            NeumannBoundaryVisitor<3, double, NeumannBoundaryType::Traction> neumann(
                [&]( const VectorXd uu ) { return vector<double>{(double)i}; } );
            cells[2]->EdgePointerGetter( 1 )->Accept( neumann );
            neumann.NeumannBoundaryAssembler( F_ext_local );
            VectorXd F_ext( dof.TotalDof() );
            F_ext.setZero();
            F_ext.segment( dof.StartingDof( cells[2]->GetID() ), domain3->GetDof() * 3 ) = F_ext_local;

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

        } while ( err > 1e-6 );
        Vector2d pos;
        pos << 1, 0;
        VectorXd res = ( domain3->CurrentConfigGetter() ).AffineMap( pos ) - domain3->AffineMap( pos );
        file << i * .1 << " " << res( 0 ) << " " << res( 1 ) << " " << res( 2 ) << endl;
    }
    file.close();
    return 0;
}