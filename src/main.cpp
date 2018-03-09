#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "MembraneStiffnessVisitor.hpp"
#include "BendingStiffnessVisitor.hpp"
#include "PostProcess.hpp"
#include "H1DomainSemiNormVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include "DofMapper.hpp"
#include <fstream>
#include <time.h>
#include <boost/math/constants/constants.hpp>
#include "BiharmonicInterfaceVisitor.hpp"
#include "StiffnessAssembler.hpp"
#include "ConstraintAssembler.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;
using GeometryVector = PhyTensorBsplineBasis<2, 3, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 3, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 3, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

int main()
{
    double nu = .0;
    double E = 4.32e8;
    double R = 25;
    double L = 50;
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(4, 0, 1);

    Vector4d
        point111(-29.282032302755088, -29.282032302755088, 29.282032302755088, 5.071796769724491),
        point112(-21.712083963243018, -30.677838684923557, 30.677838684923557, 4.795919436663918),
        point113(-14.391575887554247, -31.571002659693967, 31.571002659693967, 4.6172866417098355),
        point114(-7.195787943777123, -32.017584647079175, 32.017584647079175, 4.527970244232795),
        point115(0.0, -32.017584647079175, 32.017584647079175, 4.527970244232795),

        point121(-30.677838684923557, -21.712083963243018, 30.677838684923557, 4.795919436663918),
        point122(-22.869643776777806, -22.869643776777806, 33.02239411869588, 4.494476595178904),
        point123(-15.200184105546917, -23.632006915798854, 34.56287410349287, 4.295770397960685),
        point124(-7.600092052773459, -24.013188485309378, 35.33311409589136, 4.196417299351575),
        point125(0.0, -24.013188485309378, 35.33311409589136, 4.196417299351575),

        point131(-31.571002659693967, -14.391575887554247, 31.571002659693967, 4.6172866417098355),
        point132(-23.632006915798854, -15.200184105546917, 34.562874103492874, 4.295770397960685),
        point133(-15.739256250875362, -15.739256250875362, 36.54993607567506, 4.082178134496292),
        point134(-7.869628125437681, -16.008792323539584, 37.54346706176616, 3.975382002764096),
        point135(0.0, -16.008792323539584, 37.54346706176616, 3.975382002764096),

        point141(-32.017584647079175, -7.195787943777123, 32.017584647079175, 4.527970244232795),
        point142(-24.013188485309378, -7.600092052773459, 35.33311409589137, 4.196417299351575),
        point143(-16.008792323539584, -7.869628125437681, 37.543467061766165, 3.9753820027640954),
        point144(-8.004396161769792, -8.004396161769792, 38.64864354470356, 3.864864354470356),
        point145(0.0, -8.004396161769792, 38.64864354470356, 3.864864354470356),

        point151(-32.017584647079175, 0.0, 32.017584647079175, 4.527970244232795),
        point152(-24.013188485309378, 0.0, 35.33311409589137, 4.196417299351575),
        point153(-16.008792323539584, 0.0, 37.543467061766165, 3.9753820027640954),
        point154(-8.004396161769792, 0.0, 38.64864354470356, 3.864864354470356),
        point155(0.0, 0.0, 38.64864354470356, 3.864864354470356);

    WeightedGeometryVector points1{point111, point112, point113, point114, point115,
                                   point121, point122, point123, point124, point125,
                                   point131, point132, point133, point134, point135,
                                   point141, point142, point143, point144, point145,
                                   point151, point152, point153, point154, point155};

    Vector4d
        point211(0.0, -32.017584647079175, 32.017584647079175, 4.527970244232795),
        point212(7.195787943777123, -32.017584647079175, 32.017584647079175, 4.527970244232795),
        point213(14.391575887554247, -31.571002659693967, 31.571002659693967, 4.6172866417098355),
        point214(21.712083963243018, -30.677838684923557, 30.677838684923557, 4.795919436663918),
        point215(29.282032302755088, -29.282032302755088, 29.282032302755088, 5.071796769724491),

        point221(0.0, -24.013188485309378, 35.33311409589136, 4.196417299351575),
        point222(7.600092052773459, -24.013188485309378, 35.33311409589136, 4.196417299351575),
        point223(15.200184105546917, -23.632006915798854, 34.56287410349287, 4.295770397960685),
        point224(22.869643776777806, -22.869643776777806, 33.02239411869588, 4.494476595178904),
        point225(30.677838684923557, -21.712083963243018, 30.677838684923557, 4.795919436663918),

        point231(0.0, -16.008792323539584, 37.54346706176616, 3.975382002764096),
        point232(7.869628125437681, -16.008792323539584, 37.54346706176616, 3.975382002764096),
        point233(15.739256250875362, -15.739256250875362, 36.54993607567506, 4.082178134496292),
        point234(23.632006915798854, -15.200184105546917, 34.562874103492874, 4.295770397960685),
        point235(31.571002659693967, -14.391575887554247, 31.571002659693967, 4.6172866417098355),

        point241(0.0, -8.004396161769792, 38.64864354470356, 3.864864354470356),
        point242(8.004396161769792, -8.004396161769792, 38.64864354470356, 3.864864354470356),
        point243(16.008792323539584, -7.869628125437681, 37.543467061766165, 3.9753820027640954),
        point244(24.013188485309378, -7.600092052773459, 35.33311409589137, 4.196417299351575),
        point245(32.017584647079175, -7.195787943777123, 32.017584647079175, 4.527970244232795),

        point251(0.0, 0.0, 38.64864354470356, 3.864864354470356),
        point252(8.004396161769792, 0.0, 38.64864354470356, 3.864864354470356),
        point253(16.008792323539584, 0.0, 37.543467061766165, 3.9753820027640954),
        point254(24.013188485309378, 0.0, 35.33311409589137, 4.196417299351575),
        point255(32.017584647079175, 0.0, 32.017584647079175, 4.527970244232795);

    WeightedGeometryVector points2{point211, point212, point213, point214, point215,
                                   point221, point222, point223, point224, point225,
                                   point231, point232, point233, point234, point235,
                                   point241, point242, point243, point244, point245,
                                   point251, point252, point253, point254, point255};

    Vector4d
        point311(-32.017584647079175, 0.0, 32.017584647079175, 4.527970244232795),
        point312(-24.013188485309378, 0.0, 35.33311409589137, 4.196417299351575),
        point313(-16.008792323539584, 0.0, 37.543467061766165, 3.9753820027640954),
        point314(-8.004396161769792, 0.0, 38.64864354470356, 3.864864354470356),
        point315(0.0, 0.0, 38.64864354470356, 3.864864354470356),

        point321(-32.017584647079175, 7.195787943777123, 32.017584647079175, 4.527970244232795),
        point322(-24.013188485309378, 7.600092052773459, 35.33311409589137, 4.196417299351575),
        point323(-16.008792323539584, 7.869628125437681, 37.543467061766165, 3.9753820027640954),
        point324(-8.004396161769792, 8.004396161769792, 38.64864354470356, 3.864864354470356),
        point325(0.0, 8.004396161769792, 38.64864354470356, 3.864864354470356),

        point331(-31.571002659693967, 14.391575887554247, 31.571002659693967, 4.6172866417098355),
        point332(-23.632006915798854, 15.200184105546917, 34.562874103492874, 4.295770397960685),
        point333(-15.739256250875362, 15.739256250875362, 36.54993607567506, 4.082178134496292),
        point334(-7.869628125437681, 16.008792323539584, 37.54346706176616, 3.975382002764096),
        point335(0.0, 16.008792323539584, 37.54346706176616, 3.975382002764096),

        point341(-30.677838684923557, 21.712083963243018, 30.677838684923557, 4.795919436663918),
        point342(-22.869643776777806, 22.869643776777806, 33.02239411869588, 4.494476595178904),
        point343(-15.200184105546917, 23.632006915798854, 34.56287410349287, 4.295770397960685),
        point344(-7.600092052773459, 24.013188485309378, 35.33311409589136, 4.196417299351575),
        point345(0.0, 24.013188485309378, 35.33311409589136, 4.196417299351575),

        point351(-29.282032302755088, 29.282032302755088, 29.282032302755088, 5.071796769724491),
        point352(-21.712083963243018, 30.677838684923557, 30.677838684923557, 4.795919436663918),
        point353(-14.391575887554247, 31.571002659693967, 31.571002659693967, 4.6172866417098355),
        point354(-7.195787943777123, 32.017584647079175, 32.017584647079175, 4.527970244232795),
        point355(0.0, 32.017584647079175, 32.017584647079175, 4.527970244232795);

    WeightedGeometryVector points3{point311, point312, point313, point314, point315,
                                   point321, point322, point323, point324, point325,
                                   point331, point332, point333, point334, point335,
                                   point341, point342, point343, point344, point345,
                                   point351, point352, point353, point354, point355};

    Vector4d
        point411(0.0, 0.0, 38.64864354470356, 3.864864354470356),
        point412(8.004396161769792, 0.0, 38.64864354470356, 3.864864354470356),
        point413(16.008792323539584, 0.0, 37.543467061766165, 3.9753820027640954),
        point414(24.013188485309378, 0.0, 35.33311409589137, 4.196417299351575),
        point415(32.017584647079175, 0.0, 32.017584647079175, 4.527970244232795),

        point421(0.0, 8.004396161769792, 38.64864354470356, 3.864864354470356),
        point422(8.004396161769792, 8.004396161769792, 38.64864354470356, 3.864864354470356),
        point423(16.008792323539584, 7.869628125437681, 37.543467061766165, 3.9753820027640954),
        point424(24.013188485309378, 7.600092052773459, 35.33311409589137, 4.196417299351575),
        point425(32.017584647079175, 7.195787943777123, 32.017584647079175, 4.527970244232795),

        point431(0.0, 16.008792323539584, 37.54346706176616, 3.975382002764096),
        point432(7.869628125437681, 16.008792323539584, 37.54346706176616, 3.975382002764096),
        point433(15.739256250875362, 15.739256250875362, 36.54993607567506, 4.082178134496292),
        point434(23.632006915798854, 15.200184105546917, 34.562874103492874, 4.295770397960685),
        point435(31.571002659693967, 14.391575887554247, 31.571002659693967, 4.6172866417098355),

        point441(0.0, 24.013188485309378, 35.33311409589136, 4.196417299351575),
        point442(7.600092052773459, 24.013188485309378, 35.33311409589136, 4.196417299351575),
        point443(15.200184105546917, 23.632006915798854, 34.56287410349287, 4.295770397960685),
        point444(22.869643776777806, 22.869643776777806, 33.02239411869588, 4.494476595178904),
        point445(30.677838684923557, 21.712083963243018, 30.677838684923557, 4.795919436663918),

        point451(0.0, 32.017584647079175, 32.017584647079175, 4.527970244232795),
        point452(7.195787943777123, 32.017584647079175, 32.017584647079175, 4.527970244232795),
        point453(14.391575887554247, 31.571002659693967, 31.571002659693967, 4.6172866417098355),
        point454(21.712083963243018, 30.677838684923557, 30.677838684923557, 4.795919436663918),
        point455(29.282032302755088, 29.282032302755088, 29.282032302755088, 5.071796769724491);

    WeightedGeometryVector points4{point411, point412, point413, point414, point415,
                                   point421, point422, point423, point424, point425,
                                   point431, point432, point433, point434, point435,
                                   point441, point442, point443, point444, point445,
                                   point451, point452, point453, point454, point455};

    Matrix4d rotate_x_p, rotate_x_n, rotate_y_p, rotate_y_n;
    rotate_x_p.setZero();
    rotate_x_p(0, 0) = 1;
    rotate_x_p(1, 2) = -1;
    rotate_x_p(2, 1) = 1;
    rotate_x_p(3, 3) = 1;

    rotate_x_n.setZero();
    rotate_x_n(0, 0) = 1;
    rotate_x_n(1, 2) = 1;
    rotate_x_n(2, 1) = -1;
    rotate_x_n(3, 3) = 1;

    rotate_y_p.setZero();
    rotate_y_p(0, 2) = 1;
    rotate_y_p(1, 1) = 1;
    rotate_y_p(2, 0) = -1;
    rotate_y_p(3, 3) = 1;

    rotate_y_n.setZero();
    rotate_y_n(0, 2) = -1;
    rotate_y_n(1, 1) = 1;
    rotate_y_n(2, 0) = 1;
    rotate_y_n(3, 3) = 1;
    WeightedGeometryVector points5, points6, points7, points8, points9, points10, points11, points12;

    for (auto i : points3)
    {
        Vector4d point = rotate_x_p * i;
        points5.push_back(point);
    }
    for (auto i : points4)
    {
        Vector4d point = rotate_x_p * i;
        points6.push_back(point);
    }
    for (auto i : points1)
    {
        Vector4d point = rotate_x_n * i;
        points7.push_back(point);
    }
    for (auto i : points2)
    {
        Vector4d point = rotate_x_n * i;
        points8.push_back(point);
    }
    for (auto i : points1)
    {
        Vector4d point = rotate_y_p * i;
        points9.push_back(point);
    }
    for (auto i : points3)
    {
        Vector4d point = rotate_y_p * i;
        points10.push_back(point);
    }
    for (auto i : points2)
    {
        Vector4d point = rotate_y_n * i;
        points11.push_back(point);
    }
    for (auto i : points4)
    {
        Vector4d point = rotate_y_n * i;
        points12.push_back(point);
    }
    array<shared_ptr<PhyTensorNURBSBasis<2, 3, double>>, 12> domains;
    domains[0] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1);
    domains[1] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2);
    domains[2] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3);
    domains[3] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points4);
    domains[4] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points5);
    domains[5] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points6);
    domains[6] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points7);
    domains[7] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points8);
    domains[8] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points9);
    domains[9] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points10);
    domains[10] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points11);
    domains[11] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points12);

    int degree, refine;
    cin >> degree >> refine;
    for (auto &i : domains)
    {
        i->DegreeElevate(degree);
        i->UniformRefine(refine);
    }
    vector<shared_ptr<Surface<3, double>>> cells;
    for (int i = 0; i < 12; i++)
    {
        cells.push_back(make_shared<Surface<3, double>>(domains[i]));
        cells[i]->SurfaceInitialize();
    }
    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0, 0};
    };
    DofMapper dof;
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), 3 * i->GetDomain()->GetDof());
    }
    for (int i = 0; i < 11; i++)
    {
        for (int j = i + 1; j < 12; j++)
        {
            cells[i]->Match(cells[j]);
        }
    }

    vector<Triplet<double>> constraint;
    ConstraintAssembler<double> constraint_assemble(dof);
    auto num_of_constraints = constraint_assemble.Assemble(cells, constraint);
    auto start_index = dof.StartingDof(cells[0]->GetID());
    auto indices = cells[0]->VertexPointerGetter(2)->Indices(1, 0);
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
        num_of_constraints++;
    }
    for (auto &i : indices)
    {
        constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
        num_of_constraints++;
    }
    // indices = cells[1]->VertexPointerGetter(1)->Indices(1, 0);
    // start_index = dof.StartingDof(cells[1]->GetID());
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
    //     num_of_constraints++;
    // }
    // indices = cells[2]->VertexPointerGetter(3)->Indices(1, 0);
    // start_index = dof.StartingDof(cells[2]->GetID());
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
    //     num_of_constraints++;
    // }
    // indices = cells[3]->VertexPointerGetter(0)->Indices(1, 0);
    // start_index = dof.StartingDof(cells[3]->GetID());
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 1, 1));
    //     num_of_constraints++;
    // }
    // for (auto &i : indices)
    // {
    //     constraint.push_back(Triplet<double>(num_of_constraints, start_index + 3 * i + 2, 1));
    //     num_of_constraints++;
    // }
    SparseMatrix<double> constraint_matrix(num_of_constraints, dof.TotalDof());
    constraint_matrix.setFromTriplets(constraint.begin(), constraint.end());
    constraint_matrix.pruned(1e-13);
    constraint_matrix.makeCompressed();
    MatrixXd dense_constraint = constraint_matrix;
    FullPivLU<MatrixXd> lu_decomp(dense_constraint);
    MatrixXd x = lu_decomp.kernel();

    // ColPivHouseholderQR<MatrixXd> qr(dense_constraint);
    // // Extract Q matrix
    // MatrixXd left = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Upper>();
    // MatrixXd right = qr.matrixR().topRightCorner(qr.rank(), constraint_matrix.cols() - qr.rank());
    // FullPivLU<MatrixXd> decomp(left);
    // SparseMatrix<double, RowMajor> x(dof.TotalDof(), right.cols());

    // MatrixXd right_sol = left.colPivHouseholderQr().solve(-right);
    // x.topRows(qr.rank()) = right_sol.sparseView();
    // SparseMatrix<double, RowMajor> identity(right.cols(), right.cols());
    // identity.setIdentity();
    // x.bottomRows(right.cols()) = identity;
    // x = qr.colsPermutation() * x;
    // x.pruned(1e-10);
    StiffnessAssembler<double> stiffness_assemble(dof);
    SparseMatrix<double> stiffness_matrix, load_vector;
    stiffness_assemble.Assemble(cells, body_force, stiffness_matrix, load_vector);
    load_vector.coeffRef(dof.StartingDof(cells[6]->GetID()) + 3 * *(cells[6]->VertexPointerGetter(2)->Indices(1, 0).begin()) + 1, 0) = -2.0;
    load_vector.coeffRef(dof.StartingDof(cells[4]->GetID()) + 3 * *(cells[4]->VertexPointerGetter(3)->Indices(1, 0).begin()) + 1, 0) = 2.0;
    load_vector.coeffRef(dof.StartingDof(cells[9]->GetID()) + 3 * *(cells[9]->VertexPointerGetter(3)->Indices(1, 0).begin()), 0) = 2.0;
    load_vector.coeffRef(dof.StartingDof(cells[11]->GetID()) + 3 * *(cells[11]->VertexPointerGetter(0)->Indices(1, 0).begin()), 0) = -2.0;

    SparseMatrix<double> stiff_sol = (x.transpose() * stiffness_matrix * x).sparseView();
    stiff_sol.pruned(1e-10);
    SparseMatrix<double> load_sol = (x.transpose() * load_vector).sparseView();
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(stiff_sol);
    VectorXd solution = x * cg.solve(load_sol);

    array<ofstream, 12> myfiles;
    myfiles[0].open("domain1.txt");
    myfiles[1].open("domain2.txt");
    myfiles[2].open("domain3.txt");
    myfiles[3].open("domain4.txt");
    myfiles[4].open("domain5.txt");
    myfiles[5].open("domain6.txt");
    myfiles[6].open("domain7.txt");
    myfiles[7].open("domain8.txt");
    myfiles[8].open("domain9.txt");
    myfiles[9].open("domain10.txt");
    myfiles[10].open("domain11.txt");
    myfiles[11].open("domain12.txt");
    Vector2d u;

    array<GeometryVector, 12> solution_ctrl_pts;
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < domains[i]->GetDof(); j++)
        {
            Vector3d temp;
            temp << solution(dof.StartingDof(cells[i]->GetID()) + 3 * j + 0), solution(dof.StartingDof(cells[i]->GetID()) + 3 * j + 1), solution(dof.StartingDof(cells[i]->GetID()) + 3 * j + 2);
            solution_ctrl_pts[i].push_back(temp);
        }
    }
    for (auto &i : solution_ctrl_pts[0])
    {
        cout << i.transpose() << endl;
    }
    array<shared_ptr<PhyTensorNURBSBasis<2, 3, double>>, 12> solution_domains;
    for (int i = 0; i < 12; i++)
    {
        solution_domains[i] = make_shared<PhyTensorNURBSBasis<2, 3, double>>(std::vector<KnotVector<double>>{domains[i]->KnotVectorGetter(0), domains[i]->KnotVectorGetter(1)}, solution_ctrl_pts[i], domains[i]->WeightVectorGetter());
    }

    // // u << 1, 1;
    // // cout << solution_domain->AffineMap(u) << endl;
    // // u << 1, 0;
    // // cout << solution_domain->AffineMap(u) << endl;
    // u << 0, 1;

    VectorXd position;
    VectorXd result;
    for (int i = 0; i < 51; i++)
    {
        for (int j = 0; j < 51; j++)
        {
            u << 1.0 * i / 50, 1.0 * j / 50;
            for (int k = 0; k < 12; k++)
            {
                position = domains[k]->AffineMap(u);
                result = solution_domains[k]->AffineMap(u);
                myfiles[k] << 200 * result(0) + position(0) << " " << 200 * result(1) + position(1) << " " << 200 * result(2) + position(2) << " " << result(0) << endl;
            }
        }
    }
    // for (int i = 0; i < 12; i++)
    // {
    //     myfiles[i].close();
    // }
    return 0;
}