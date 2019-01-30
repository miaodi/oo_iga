
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
    Vector2d point2( 0, 1.0 );
    Vector2d point3( 1.0, 0 );
    Vector2d point4( 1.0, 1.0 );

    GeometryVector points1( {point1, point2, point3, point4} );

    shared_ptr<PhyTensorBsplineBasis<2, 2, double>> domain =
        make_shared<PhyTensorBsplineBasis<2, 2, double>>( std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1 );

    domain->DegreeElevate( 2 );
    domain->UniformRefine( 1 );
    int dof = domain->GetDof();
    VectorXd c = VectorXd::Random( dof ) * .05 + VectorXd::Constant( dof, .63 );
    VectorXd ct = VectorXd::Zero( dof );
    shared_ptr<Surface<2, double>> cell = make_shared<Surface<2, double>>( domain );
    cell->SurfaceInitialize();
    auto load = []( const Vector2d& u ) { return std::vector<double>{0, 0}; };
    CH4thStiffnessVisitor<double> CH4thsv( load );
    CH2ndStiffnessVisitor<double> CH2ndsv( load );
    CHMassVisitor<double> CHmv( load );
    CH4thsv.SetStateData( c.data() );
    CH2ndsv.SetStateData( c.data() );
    CHmv.SetStateData( ct.data() );

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
    cout << MatrixXd( stiffness_matrix_ch4th ) << endl;
    cout << MatrixXd( load_vector_ch4th ) << endl;

    const auto stiffness_triplet_ch2nd = CH2ndsv.GetStiffness();
    const auto load_triplet_ch2nd = CH2ndsv.GetRhs();
    SparseMatrix<double> stiffness_matrix_ch2nd, load_vector_ch2nd;
    load_vector_ch2nd.resize( dof, 1 );
    stiffness_matrix_ch2nd.resize( dof, dof );
    load_vector_ch2nd.setFromTriplets( load_triplet_ch2nd.begin(), load_triplet_ch2nd.end() );
    stiffness_matrix_ch2nd.setFromTriplets( stiffness_triplet_ch2nd.begin(), stiffness_triplet_ch2nd.end() );
    cout << MatrixXd( stiffness_matrix_ch2nd ) << endl;
    cout << MatrixXd( load_vector_ch2nd ) << endl;

    const auto stiffness_triplet_chm = CHmv.GetStiffness();
    const auto load_triplet_chm = CHmv.GetRhs();
    SparseMatrix<double> stiffness_matrix_chm, load_vector_chm;
    load_vector_chm.resize( dof, 1 );
    stiffness_matrix_chm.resize( dof, dof );
    load_vector_chm.setFromTriplets( load_triplet_chm.begin(), load_triplet_chm.end() );
    stiffness_matrix_chm.setFromTriplets( stiffness_triplet_chm.begin(), stiffness_triplet_chm.end() );
    cout << MatrixXd( stiffness_matrix_chm ) << endl;
    cout << MatrixXd( load_vector_chm ) << endl;
    cout << ct << endl;
    return 0;
}