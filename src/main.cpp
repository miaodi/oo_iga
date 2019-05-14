
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
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 1, 0, 1 );
    Vector3d point1( 0, 0, 0 ), point2( 0, 1, 0 ), point3( 10, 0, 0 ), point4( 10, 1, 0 );
    GeometryVector points{point1, point2, point3, point4};

    auto domain =
        make_shared<PhyTensorBsplineBasis<2, 3, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points );
    int degree, refine;
    cin >> degree >> refine;
    domain->DegreeElevate( degree );
    domain->KnotsInsertion( 0, {.1, .2, .3, .4, .5, .6, .7, .8, .9} );
    domain->UniformRefine( refine );

    domain->CreateCurrentConfig();
    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain ) );
    cells[0]->SurfaceInitialize();
    function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
        return vector<double>{0, 0, 0};
    };
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }
    VectorXd u( dof.TotalDof() );

    vector<int> boundary_indices;
    boundary_indices = cells[0]->EdgePointerGetter( 3 )->Indices( 3, 1 );

    KLShellConstraintAssembler<double> ca( dof );
    ca.ConstraintInitialize( cells );
    ca.ConstraintCreator( cells );
    ca.Additional_Constraint( boundary_indices );
    SparseMatrix<double> constraint_basis;
    ca.AssembleConstraints( constraint_basis );
    for ( int i = 1; i <= 100; i++ )
    {
        double err;
        double init_err = 0;
        do
        {
            SparseMatrix<double> stiffness_matrix_bend, stiffness_matrix_mem, F_int_bend, F_int_mem, F_ext;
            stiffness_matrix_bend.resize( dof.TotalDof(), dof.TotalDof() );
            stiffness_matrix_mem.resize( dof.TotalDof(), dof.TotalDof() );
            F_int_bend.resize( dof.TotalDof(), 1 );
            F_int_mem.resize( dof.TotalDof(), 1 );
            F_ext.resize( dof.TotalDof(), 1 );
            StiffnessAssembler<NonlinearBendingStiffnessVisitor<double>> bending_stiffness_assemble( dof );
            bending_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_bend, F_int_bend );
            StiffnessAssembler<NonlinearMembraneStiffnessVisitor<double>> membrane_stiffness_assemble( dof );
            membrane_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_mem, F_int_mem );
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_bend + stiffness_matrix_mem;
            cout << "Constructed stiffness matrix.\n";
            NeumannBoundaryVisitor<3, double, NeumannBoundaryType::Traction> neumann(
                [&]( const VectorXd uu ) { return vector<double>{(double)i}; } );
            cells[0]->EdgePointerGetter( 1 )->Accept( neumann );
            neumann.NeumannBoundaryAssembler( F_ext );

            SparseMatrix<double> F = F_ext - F_int_mem - F_int_bend;
            cout << "Constructed load vector.\n";
            SparseMatrix<double> stiff_sol = ( constraint_basis.transpose() * stiffness_matrix * constraint_basis );
            SparseMatrix<double> load_sol = ( constraint_basis.transpose() * F );
            ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
            cg.compute( stiff_sol );
            u = constraint_basis * cg.solve( load_sol );
            cout << "Solved the problem.\n";
            // cout << u.transpose() << endl;
            Map<MatrixXd> umat( u.data(), 3, u.size() / 3 );
            err = load_sol.norm();

            if ( init_err == 0 )
                init_err = err;
            err /= init_err;
            cout << err << endl;
            domain->UpdateCurrentGeometryVector( umat );
        } while ( err > 1e-5 );
        domain->UpdateGeometryVector();
    }

    Vector2d pos;
    pos << 1, .5;
    cout << ( domain->CurrentConfigGetter() ).AffineMap( pos );

    // GeometryVector solution_ctrl_pts1, solution_ctrl_pts2, solution_ctrl_pts3, solution_ctrl_pts4;
    // for ( int i = 0; i < domain1->GetDof(); i++ )
    // {
    //     Vector3d temp;
    //     temp << solution( 3 * i + 0 ), solution( 3 * i + 1 ), solution( 3 * i + 2 );
    //     solution_ctrl_pts1.push_back( temp );
    // }
    // for ( int i = 0; i < domain2->GetDof(); i++ )
    // {
    //     Vector3d temp;
    //     temp << solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 0 ),
    //         solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 1 ),
    //         solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 2 );
    //     solution_ctrl_pts2.push_back( temp );
    // }

    // auto solution_domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
    //     std::vector<KnotVector<double>>{domain1->KnotVectorGetter( 0 ), domain1->KnotVectorGetter( 1 )},
    //     solution_ctrl_pts1, domain1->WeightVectorGetter() );
    // auto solution_domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
    //     std::vector<KnotVector<double>>{domain2->KnotVectorGetter( 0 ), domain2->KnotVectorGetter( 1 )},
    //     solution_ctrl_pts2, domain2->WeightVectorGetter() );
    // Vector2d u;
    // u << 0, 1;
    // cout << setprecision( 10 ) << solution_domain1->AffineMap( u ) << std::endl;
    // cout << dof.TotalDof() << endl;
    // cout << abs( solution_domain1->AffineMap( u )( 2 ) + 0.300592457 ) / 0.300592457 << endl;
    return 0;
}