
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
    double nu = .0;
    double E = 4.32e8;
    double R = 25;
    double L = 50;
    KnotVector<double> knot_vector;
    knot_vector.InitClosed( 2, 0, 1 );
    double rad = 40.0 / 180 * boost::math::constants::pi<double>();
    double a = sin( rad ) * R;
    double b = a * tan( rad );
    Vector3d point1( -a, 0, 0 ), point2( -a, L / 2, 0 ), point3( -a, L / 1, 0 ), point4( 0, 0, b ),
        point5( 0, L / 2, b ), point6( 0, L / 1, b ), point7( a, 0, 0 ), point8( a, L / 2, 0 ), point9( a, L / 1, 0 );
    GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};

    Vector1d weight1( 1 ), weight2( sin( boost::math::constants::pi<double>() / 2 - rad ) );
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};

    int degree, refine;
    cin >> degree >> refine;
    for ( int d = 0; d < degree; ++d )
    {
        for ( int r = 0; r < refine; ++r )
        {
            auto domain = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
                std::vector<KnotVector<double>>{knot_vector, knot_vector}, points, weights );

            domain->DegreeElevate( d );
            domain->UniformRefine( r );
            vector<shared_ptr<Surface<3, double>>> cells;
            cells.push_back( make_shared<Surface<3, double>>( domain ) );
            cells[0]->SurfaceInitialize();
            function<vector<double>( const VectorXd& )> body_force = []( const VectorXd& u ) {
                return vector<double>{0, 0, -90};
            };
            DofMapper dof;
            for ( auto& i : cells )
            {
                dof.Insert( i->GetID(), 3 * i->GetDomain()->GetDof() );
            }

            vector<int> boundary_indices;
            auto indices = cells[0]->EdgePointerGetter( 0 )->Indices( 1, 0 );
            for ( auto& i : indices )
            {
                boundary_indices.push_back( 3 * i );
            }
            for ( auto& i : indices )
            {
                boundary_indices.push_back( 3 * i + 2 );
            }
            boundary_indices.push_back( 1 );

            indices = cells[0]->EdgePointerGetter( 2 )->Indices( 1, 0 );
            for ( auto& i : indices )
            {
                boundary_indices.push_back( 3 * i );
            }
            for ( auto& i : indices )
            {
                boundary_indices.push_back( 3 * i + 2 );
            }
            sort( boundary_indices.begin(), boundary_indices.end() );
            boundary_indices.erase( unique( boundary_indices.begin(), boundary_indices.end() ), boundary_indices.end() );

            KLShellConstraintAssembler<double> ca( dof );
            ca.ConstraintInitialize( cells );
            // ConstraintAssembler<2, 3, double> ca( dof );

            ca.ConstraintCreator( cells );
            ca.Additional_Constraint( boundary_indices );
            SparseMatrix<double> constraint_basis;
            ca.AssembleConstraints( constraint_basis );

            StiffnessAssembler<BendingStiffnessVisitor<double>> bending_stiffness_assemble( dof );
            SparseMatrix<double> stiffness_matrix_bend, stiffness_matrix_mem, load_vector;
            stiffness_matrix_bend.resize( dof.TotalDof(), dof.TotalDof() );
            stiffness_matrix_mem.resize( dof.TotalDof(), dof.TotalDof() );
            load_vector.resize( dof.TotalDof(), 1 );
            bending_stiffness_assemble.Assemble( cells, body_force, stiffness_matrix_bend, load_vector );
            StiffnessAssembler<MembraneStiffnessVisitor<double>> membrane_stiffness_assemble( dof );
            membrane_stiffness_assemble.Assemble( cells, stiffness_matrix_mem );
            SparseMatrix<double> stiffness_matrix = stiffness_matrix_bend + stiffness_matrix_mem;

            SparseMatrix<double> stiff_sol = ( constraint_basis.transpose() * stiffness_matrix * constraint_basis );
            SparseMatrix<double> load_sol = ( constraint_basis.transpose() * load_vector );
            ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
            cg.compute( stiff_sol );
            VectorXd solution = constraint_basis * cg.solve( load_sol );

            GeometryVector solution_ctrl_pts;
            for ( int i = 0; i < domain->GetDof(); i++ )
            {
                Vector3d temp;
                temp << solution( 3 * i + 0 ), solution( 3 * i + 1 ), solution( 3 * i + 2 );
                solution_ctrl_pts.push_back( temp );
            }

            auto solution_domain = make_shared<PhyTensorNURBSBasis<2, 3, double>>(
                std::vector<KnotVector<double>>{domain->KnotVectorGetter( 0 ), domain->KnotVectorGetter( 1 )},
                solution_ctrl_pts, domain->WeightVectorGetter() );
            Vector2d u;
            u << 0, .5;
            cout << setprecision( 12 ) << solution_domain->AffineMap( u ) << std::endl;
            cout << dof.TotalDof() << endl;
            cout << abs( solution_domain->AffineMap( u )( 2 ) + 0.300592457 ) / 0.300592457 << endl;
        }
        cout << endl;
    }
    return 0;
}