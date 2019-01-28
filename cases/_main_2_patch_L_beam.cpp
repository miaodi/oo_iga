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
    knot_vector.InitClosed( 1, 0, 1 );

    Vector3d point11( 0.0, 0.0, 0.0 ), point12( 0.0, 10.0, 0.0 ), point13( 2.0, 0.0, 0.0 ),
        point14( 2.0, 10.0, 0.0 );
    Vector3d point21( 0.0, 0.0, 0.0 ), point22( 0.0, 10.0, 0.0 ), point23( 0.0, 0.0, -2.0 ), point24( 0.0, 10.0, -2.0 );
    GeometryVector points1{point11, point12, point13, point14};
    GeometryVector points2{point21, point22, point23, point24};
    auto domain1 =
        make_shared<PhyTensorBsplineBasis<2, 3, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );
    auto domain2 =
        make_shared<PhyTensorBsplineBasis<2, 3, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2 );

    int degree, refine;
    cin >> degree >> refine;

    domain1->DegreeElevate( degree );
    domain2->DegreeElevate( degree );

    domain1->KnotsInsertion( 1, {1.0 / 4, 2.0 / 4, 3.0 / 4} );
    domain2->KnotsInsertion( 1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5} );

    domain1->UniformRefine( refine );
    domain2->UniformRefine( refine );

    vector<shared_ptr<Surface<3, double>>> cells;
    cells.push_back( make_shared<Surface<3, double>>( domain1 ) );
    cells[0]->SurfaceInitialize();
    cells.push_back( make_shared<Surface<3, double>>( domain2 ) );
    cells[1]->SurfaceInitialize();
    DofMapper dof;
    for ( auto& i : cells )
    {
        dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
    }
    for ( int i = 0; i < 1; i++ )
    {
        for ( int j = i + 1; j < 2; j++ )
        {
            cells[i]->Match( cells[j] );
        }
    }
    StiffnessAssembler<BendingStiffnessVisitor<double>> bending_stiffness_assemble( dof );
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_matrix.resize( dof.TotalDof(), dof.TotalDof() );
    load_vector.resize( dof.TotalDof(), 1 );
    bending_stiffness_assemble.Assemble( cells, stiffness_matrix );
    StiffnessAssembler<MembraneStiffnessVisitor<double>> membrane_stiffness_assemble( dof );
    membrane_stiffness_assemble.Assemble( cells, stiffness_matrix );

    vector<int> boundary_indices;
    auto indices = cells[0]->EdgePointerGetter( 0 )->Indices( 1, 1 );
    auto start_index = dof.StartingDof( cells[0]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 1 );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    indices = cells[1]->EdgePointerGetter( 0 )->Indices( 1, 1 );
    start_index = dof.StartingDof( cells[1]->GetID() );
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 1 );
    }
    for ( auto& i : indices )
    {
        boundary_indices.push_back( start_index + 3 * i + 2 );
    }
    sort( boundary_indices.begin(), boundary_indices.end() );
    boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

    indices = cells[0]->VertexPointerGetter( 2 )->Indices( 1, 0 );
    load_vector.coeffRef( 3 * indices[0] + 2, 0 ) = -1;

    ConstraintAssembler<2, 3, double> constraint_assemble( dof );
    constraint_assemble.ConstraintCreator( cells );
    constraint_assemble.Additional_Constraint( boundary_indices );

    SparseMatrix<double, RowMajor> constraint_matrix;
    constraint_assemble.AssembleConstraintWithAdditionalConstraint( constraint_matrix );

    constraint_matrix.makeCompressed();
    MatrixXd dense_constraint = constraint_matrix;

    FullPivLU<MatrixXd> lu_decomp( dense_constraint );
    lu_decomp.setThreshold( 6e-10 );
    MatrixXd x = lu_decomp.kernel();
    SparseMatrix<double> stiff_sol = ( x.transpose() * stiffness_matrix * x ).sparseView();

    SparseMatrix<double> load_sol = ( x.transpose() * load_vector ).sparseView();
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute( stiff_sol );
    VectorXd solution = x * cg.solve( load_sol );

    GeometryVector solution_ctrl_pts1, solution_ctrl_pts2;
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

    for ( int i = 0; i < domain1->GetDof(); i++ )
    {
        Vector3d temp = solution_ctrl_pts1[i] + domain1->CtrPtsGetter( i );
        solution_ctrl_pts1[i] = temp;
        // cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }
    cout << endl;
    for ( int i = 0; i < domain2->GetDof(); i++ )
    {
        Vector3d temp = solution_ctrl_pts2[i] + domain2->CtrPtsGetter( i );
        solution_ctrl_pts2[i] = temp;
        // cout << "(" << temp( 0 ) << ", " << temp( 1 ) << ", " << temp( 2 ) << ")," << endl;
    }

    auto solution_domain1 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{domain1->KnotVectorGetter( 0 ), domain1->KnotVectorGetter( 1 )}, solution_ctrl_pts1 );
    auto solution_domain2 = make_shared<PhyTensorBsplineBasis<2, 3, double>>(
        std::vector<KnotVector<double>>{domain2->KnotVectorGetter( 0 ), domain2->KnotVectorGetter( 1 )}, solution_ctrl_pts2 );
    Vector2d u;

    for ( int i = 0; i <= 100; i++ )
    {
        u << 0, 1.0 * i / 100;
        MatrixXd j1 = domain1->JacobianMatrix( u );
        MatrixXd j2 = domain2->JacobianMatrix( u );
        MatrixXd jd1 = solution_domain1->JacobianMatrix( u );
        MatrixXd jd2 = solution_domain2->JacobianMatrix( u );
        Vector3d n1 = Vector3d( j1.col( 0 ) ).cross( Vector3d( j1.col( 1 ) ) ).normalized();
        Vector3d n2 = Vector3d( j2.col( 0 ) ).cross( Vector3d( j2.col( 1 ) ) ).normalized();
        Vector3d nd1 = Vector3d( jd1.col( 0 ) ).cross( Vector3d( jd1.col( 1 ) ) ).normalized();
        Vector3d nd2 = Vector3d( jd2.col( 0 ) ).cross( Vector3d( jd2.col( 1 ) ) ).normalized();
        double d = acos( n2.dot( n1 ) ) * 180 / Pi;
        double dd = acos( nd2.dot( nd1 ) ) * 180 / Pi;
        cout << domain1->AffineMap( u )( 1 ) << " " << abs( d - dd ) / abs( d ) << endl;
    }
    u << 1, .25;
    cout << domain1->AffineMap( u ) << endl;
    u << 1, .5;
    cout << domain1->AffineMap( u ) << endl;
    u << 1, .75;
    cout << domain1->AffineMap( u ) << endl;
    u << 1, 1;
    cout << setprecision( 10 ) << solution_domain1->AffineMap( u ) << std::endl;
    // cout << setprecision( 10 ) << solution_domain2->JacobianMatrix( u ) << std::endl;

    return 0;
}