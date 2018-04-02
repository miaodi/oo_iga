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
    KnotVector<double> knot_vector;
    knot_vector.InitClosed(3, 0, 1);
    knot_vector.UniformRefine(3);
    knot_vector.printKnotVector();
    BsplineBasis<double> a(knot_vector);
    a.ModifyBoundaryInitialize();
    ofstream myfile;
    myfile.open("basis_reduce_degree.txt");
    auto dof = a.GetDof() - 4;
    myfile << "x ";
    for (int i = 0; i < dof; i++)
    {
        myfile << "N" << i << " ";
    }
    myfile << endl;
    for (int i = 0; i < 201; i++)
    {
        double u = i * 1.0 / 200;
        myfile << u << " ";
        auto res = a.EvalModifiedDerAll(u, 0);
        vector<double> values(dof, 0);
        for (auto &i : *res)
        {
            values[i.first] = i.second[0];
        }
        for (auto &i : values)
        {
            myfile << setprecision(4) << i << " ";
        }
        myfile << endl;
    }
    myfile.close();
    ofstream myfile1;
    myfile1.open("basis_original.txt");

    auto dof1 = a.GetDof();
    myfile1 << "x ";
    for (int i = 0; i < dof1; i++)
    {
        myfile1 << "N" << i << " ";
    }
    myfile1 << endl;
    for (int i = 0; i < 201; i++)
    {
        double u = i * 1.0 / 200;
        myfile1 << u << " ";
        auto res = a.EvalDerAll(u, 0);
        vector<double> values(dof1, 0);
        for (auto &i : *res)
        {
            values[i.first] = i.second[0];
        }
        for (auto &i : values)
        {
            myfile1 << setprecision(4) << i << " ";
        }
        myfile1 << endl;
    }
    myfile1.close();

    ofstream myfile2;
    myfile2.open("basis_coarse.txt");
    myfile2 << "x ";
    for (int i = 0; i < dof; i++)
    {
        myfile2 << "N" << i << " ";
    }
    myfile2 << endl;
    for (int i = 0; i < 201; i++)
    {
        double u = i * 1.0 / 200;
        myfile2 << u << " ";
        auto res = a.EvalDerAll(u, 0);
        vector<double> values1(dof1, 0);
        for (auto &i : *res)
        {
            values1[i.first] = i.second[0];
        }
        vector<double> values(dof, 0);
        int k = 0;
        for (auto &i : values)
        {
            i = values1[k + 2];
            k++;
        }
        values[0] = values1[0] + values1[1] + values1[2];
        *values.rbegin() = *values1.rbegin() + *(values1.rbegin() + 1) + *(values1.rbegin() + 2);
        for (auto &i : values)
        {
            myfile2 << setprecision(4) << i << " ";
        }
        myfile2 << endl;
    }
    myfile2.close();
    return 0;
}