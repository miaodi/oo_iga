#include "BendingStiffnessVisitor.hpp"
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
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/KroneckerProduct>
#include <fstream>
#include <iostream>
#include <time.h>

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
    knot_vector.InitClosed( 2, 0, 1 );

    Vector3d point11( -16.06969024216348, 0.0, 0.0 ), point12( -16.06969024216348, 12.5, 0.0 ),
        point13( -16.06969024216348, 25.0, 0.0 ), point14( -9.09925585665506, 0.0, 5.84888892202555 ),
        point15( -9.09925585665506, 12.5, 5.84888892202555 ), point16( -9.09925585665506, 25.0, 5.84888892202555 ),
        point17( 0.0, 0.0, 5.84888892202555 ), point18( 0.0, 12.5, 5.84888892202555 ), point19( 0.0, 25.0, 5.84888892202555 );
    Vector3d point21( -16.06969024216348, 25.0, 0.0 ), point22( -16.06969024216348, 37.5, 0.0 ),
        point23( -16.06969024216348, 50.0, 0.0 ), point24( -9.09925585665506, 25.0, 5.84888892202555 ),
        point25( -9.09925585665506, 37.5, 5.84888892202555 ), point26( -9.09925585665506, 50.0, 5.84888892202555 ),
        point27( 0.0, 25.0, 5.84888892202555 ), point28( 0.0, 37.5, 5.84888892202555 ), point29( 0.0, 50.0, 5.84888892202555 );
    Vector3d point31( 0.0, 0.0, 5.84888892202555 ), point32( 0.0, 12.5, 5.84888892202555 ),
        point33( 0.0, 25.0, 5.84888892202555 ), point34( 9.09925585665506, 0.0, 5.84888892202555 ),
        point35( 9.09925585665506, 12.5, 5.84888892202555 ), point36( 9.09925585665506, 25.0, 5.84888892202555 ),
        point37( 16.06969024216348, 0.0, 0.0 ), point38( 16.06969024216348, 12.5, 0.0 ),
        point39( 16.06969024216348, 25.0, 0.0 );
    Vector3d point41( 0.0, 25.0, 5.84888892202555 ), point42( 0.0, 37.5, 5.84888892202555 ),
        point43( 0.0, 50.0, 5.84888892202555 ), point44( 9.09925585665506, 25.0, 5.84888892202555 ),
        point45( 9.09925585665506, 37.5, 5.84888892202555 ), point46( 9.09925585665506, 50.0, 5.84888892202555 ),
        point47( 16.06969024216348, 25.0, 0.0 ), point48( 16.06969024216348, 37.5, 0.0 ),
        point49( 16.06969024216348, 50.0, 0.0 );
    GeometryVector points1{point11, point12, point13, point14, point15, point16, point17, point18, point19};
    GeometryVector points2{point21, point22, point23, point24, point25, point26, point27, point28, point29};
    GeometryVector points3{point31, point32, point33, point34, point35, point36, point37, point38, point39};
    GeometryVector points4{point41, point42, point43, point44, point45, point46, point47, point48, point49};
    Vector1d weight1( 1 ), weight2( 0.883022221559489 );
    WeightVector weights1{weight1, weight1, weight1, weight2, weight2, weight2, weight2, weight2, weight2};
    WeightVector weights2{weight1, weight1, weight1, weight2, weight2, weight2, weight2, weight2, weight2};
    WeightVector weights3{weight2, weight2, weight2, weight2, weight2, weight2, weight1, weight1, weight1};
    WeightVector weights4{weight2, weight2, weight2, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1, weights1 );
    auto domain2 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2, weights2 );
    auto domain3 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3, weights3 );
    auto domain4 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4, weights4 );

    int degree, refine;
    cin >> degree >> refine;
    domain1->DegreeElevate( degree );
    domain2->DegreeElevate( degree );
    domain3->DegreeElevate( degree );
    domain4->DegreeElevate( degree );

    // domain1->KnotInsertion( 0, .5 );
    // domain2->KnotInsertion( 0, 1.0 / 3 );
    // domain2->KnotInsertion( 0, 2.0 / 3 );
    // domain1->KnotInsertion( 1, .5 );
    // domain2->KnotInsertion( 1, 1.0 / 3 );
    // domain2->KnotInsertion( 1, 2.0 / 3 );
    // domain4->KnotInsertion( 0, .5 );
    // domain3->KnotInsertion( 0, 1.0 / 3 );
    // domain3->KnotInsertion( 0, 2.0 / 3 );
    // domain4->KnotInsertion( 1, .5 );
    // domain3->KnotInsertion( 1, 1.0 / 3 );
    // domain3->KnotInsertion( 1, 2.0 / 3 );

    domain1->UniformRefine( refine );
    domain2->UniformRefine( refine );
    domain3->UniformRefine( refine );
    domain4->UniformRefine( refine );
    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain1 ) );
    cells[0]->SurfaceInitialize();
    cells.push_back( make_shared<Surface<3, double>>( domain2 ) );
    cells[1]->SurfaceInitialize();
    cells.push_back( make_shared<Surface<3, double>>( domain3 ) );
    cells[2]->SurfaceInitialize();
    cells.push_back( make_shared<Surface<3, double>>( domain4 ) );
    cells[3]->SurfaceInitialize();
    function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
        return vector<double>{0, 0, -90};
    };
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }
    for ( int i = 0; i < 3; i++ )
    {
        for ( int j = i + 1; j < 4; j++ )
        {
            cells[i]->Match( cells[j] );
        }
    }
    StiffnessAssembler<BendingStiffnessVisitor<double>> bending_stiffness_assemble( dof );
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_matrix.resize( dof.TotalDof(), dof.TotalDof() );
    load_vector.resize( dof.TotalDof(), 1 );
    bending_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix, load_vector );
    StiffnessAssembler<MembraneStiffnessVisitor<double>> membrane_stiffness_assemble( dof );
    membrane_stiffness_assemble.Assemble( cells, stiffness_matrix );

    vector<int> boundary_indices;
    auto indices = cells[0]->EdgePointerGetter( 0 )->Indices( 1, 0 );
    auto start_index = dof.StartingDof( cells[0]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    boundary_indices.push_back( start_index + 1 );

    indices = cells[1]->EdgePointerGetter( 2 )->Indices( 1, 0 );
    start_index = dof.StartingDof( cells[1]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    indices = cells[2]->EdgePointerGetter( 0 )->Indices( 1, 0 );
    start_index = dof.StartingDof( cells[2]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    indices = cells[3]->EdgePointerGetter( 2 )->Indices( 1, 0 );
    start_index = dof.StartingDof( cells[3]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    sort( boundary_indices.begin(), boundary_indices.end() );
    boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

    ConstraintAssembler<2, 3, double> constraint_assemble( dof );
    constraint_assemble.ConstraintCreator( cells );
    constraint_assemble.Additional_Constraint( boundary_indices );

    SparseMatrix<double, RowMajor> constraint_matrix;
    constraint_assemble.AssembleConstraintWithAdditionalConstraint( constraint_matrix );
    constraint_matrix.pruned( 1e-10 );
    constraint_matrix.makeCompressed();
    MatrixXd dense_constraint = constraint_matrix;

    FullPivLU<MatrixXd> lu_decomp( dense_constraint );
    MatrixXd x = lu_decomp.kernel();
    SparseMatrix<double> stiff_sol = ( x.transpose() * stiffness_matrix * x ).sparseView();
    stiff_sol.pruned( 1e-10 );
    SparseMatrix<double> load_sol = ( x.transpose() * load_vector ).sparseView();
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute( stiff_sol );
    VectorXd solution = x * cg.solve( load_sol );

    GeometryVector solution_ctrl_pts1, solution_ctrl_pts2, solution_ctrl_pts3, solution_ctrl_pts4;
    for ( int i = 0; i < domain1->GetDof(); i++ )
    {
        Vector3d temp;
        temp << solution( 3 * i + 0 ), solution( 3 * i + 1 ), solution( 3 * i + 2 );
        solution_ctrl_pts1.push_back( temp );
    }
    for ( int i = 0; i < domain2->GetDof(); i++ )
    {
        Vector3d temp;
        temp << solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 0 ),
            solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 1 ),
            solution( dof.StartingDof( cells[1]->GetID() ) + 3 * i + 2 );
        solution_ctrl_pts2.push_back( temp );
    }
    for ( int i = 0; i < domain3->GetDof(); i++ )
    {
        Vector3d temp;
        temp << solution( dof.StartingDof( cells[2]->GetID() ) + 3 * i + 0 ),
            solution( dof.StartingDof( cells[2]->GetID() ) + 3 * i + 1 ),
            solution( dof.StartingDof( cells[2]->GetID() ) + 3 * i + 2 );
        solution_ctrl_pts3.push_back( temp );
    }
    for ( int i = 0; i < domain4->GetDof(); i++ )
    {
        Vector3d temp;
        temp << solution( dof.StartingDof( cells[3]->GetID() ) + 3 * i + 0 ),
            solution( dof.StartingDof( cells[3]->GetID() ) + 3 * i + 1 ),
            solution( dof.StartingDof( cells[3]->GetID() ) + 3 * i + 2 );
        solution_ctrl_pts4.push_back( temp );
    }
    for ( int i = 0; i < domain1->GetDof(); i++ )
    {
        Vector3d temp = 20 * solution_ctrl_pts1[i] + domain1->CtrPtsGetter( i );
        cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }
    domain1->PrintWtCtrPts();
    cout << endl;
    for ( int i = 0; i < domain2->GetDof(); i++ )
    {
        Vector3d temp = 20 * solution_ctrl_pts2[i] + domain2->CtrPtsGetter( i );
        cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }
    domain2->PrintWtCtrPts();
    cout << endl;
    for ( int i = 0; i < domain3->GetDof(); i++ )
    {
        Vector3d temp = 20 * solution_ctrl_pts3[i] + domain3->CtrPtsGetter( i );
        cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }
    domain3->PrintWtCtrPts();
    cout << endl;
    for ( int i = 0; i < domain4->GetDof(); i++ )
    {
        Vector3d temp = 20 * solution_ctrl_pts4[i] + domain4->CtrPtsGetter( i );
        cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }
    domain4->PrintWtCtrPts();
    cout << endl;
    auto solution_domain1 = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{domain1->KnotVectorGetter( 0 ), domain1->KnotVectorGetter( 1 )},
        solution_ctrl_pts1, domain1->WeightVectorGetter() );
    Vector2d u;
    u << 0, 1;
    cout << setprecision( 10 ) << solution_domain1->AffineMap( u ) << std::endl;
    cout << dof.TotalDof() << endl;
    cout << abs( solution_domain1->AffineMap( u )( 2 ) + 0.300592457 ) / 0.300592457 << endl;
    return 0;
}