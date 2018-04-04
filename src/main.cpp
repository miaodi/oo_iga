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
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using WeightedGeometryVector = PhyTensorNURBSBasis<2, 2, double>::WeightedGeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, double>::WeightVector;
using Vector1d = Matrix<double, 1, 1>;

int main()
{
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(2, 0, 1);

    Vector2d point11(-2.7, -.7), point12(-2.2, -.1), point13(-2, .9), point14(-1.3, -1), point15(-1.3, -.5), point16(-.7, .6), point17(-.4, -.9), point18(.1, -.3), point19(0, 0);
    Vector2d point21(-.4, -.9), point22(.1, -.3), point23(0, 0), point24(1.2, -.8), point25(1.0, .1), point26(.8, 1.2), point27(2.5, -1), point28(2.3, -.5), point29(1.9, 1.8);
    Vector2d point31(-2, .9), point32(-1.3, 1.2), point33(.1, 2.1), point34(-.7, .6), point35(-.1, 1.3), point36(1.2, 2), point37(0, 0), point38(.8, 1.2), point39(1.9, 1.8);

    GeometryVector points1{point11, point12, point13, point14, point15, point16, point17, point18, point19};
    GeometryVector points2{point21, point22, point23, point24, point25, point26, point27, point28, point29};
    GeometryVector points3{point31, point32, point33, point34, point35, point36, point37, point38, point39};

    array<shared_ptr<PhyTensorBsplineBasis<2, 2, double>>, 3> domains;
    domains[0] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points1);
    domains[1] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points2);
    domains[2] = make_shared<PhyTensorBsplineBasis<2, 2, double>>(std::vector<KnotVector<double>>{knot_vector, knot_vector}, points3);

    int degree, refine;
    cin >> degree >> refine;
    for (auto &i : domains)
    {
        i->DegreeElevate(degree);
        i->UniformRefine(refine);
    }
    domains[0]->KnotsInsertion(0, {1.0 / 3, 2.0 / 3});
    domains[0]->KnotsInsertion(1, {1.0 / 3, 2.0 / 3});
    domains[1]->KnotsInsertion(0, {1.0 / 2});
    domains[1]->KnotsInsertion(1, {1.0 / 2});
    domains[2]->KnotsInsertion(0, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5});
    domains[2]->KnotsInsertion(1, {1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5});

    vector<shared_ptr<Surface<2, double>>> cells;
    for (int i = 0; i < 3; i++)
    {
        cells.push_back(make_shared<Surface<2, double>>(domains[i]));
        cells[i]->SurfaceInitialize();
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i + 1; j < 3; j++)
        {
            cells[i]->Match(cells[j]);
        }
    }
    DofMapper dof;
    for (auto &i : cells)
    {
        dof.Insert(i->GetID(), i->GetDomain()->GetDof());
    }
    ConstraintAssembler<2, 2, double> constraint_assemble(dof);
    vector<Triplet<double>> constraint;
    auto num_of_constraints = constraint_assemble.AssembleByReducedKernel(cells, constraint);

    return 0;
}