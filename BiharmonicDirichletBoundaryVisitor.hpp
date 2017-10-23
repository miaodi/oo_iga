//
// Created by di miao on 10/23/17.
//

#pragma once

#include "DirichletBoundaryVisitor.hpp"
#include "DofMapper.hpp"

template<int N, typename T>
class BiharmonicDirichletBoundaryVisitor : public DirichletBoundaryVisitor<N, T>
{
    using Knot = typename DirichletBoundaryVisitor<N, T>::Knot;
    using Quadrature = typename DirichletBoundaryVisitor<N, T>::Quadrature;
    using QuadList = typename DirichletBoundaryVisitor<N, T>::QuadList;
    using KnotSpan = typename DirichletBoundaryVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename DirichletBoundaryVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename DirichletBoundaryVisitor<N, T>::LoadFunctor;
    using Matrix = typename DirichletBoundaryVisitor<N, T>::Matrix;
    using Vector = typename DirichletBoundaryVisitor<N, T>::Vector;
    using DomainShared_ptr = typename DirichletBoundaryVisitor<N, T>::DomainShared_ptr;

public:
    BiharmonicDirichletBoundaryVisitor(const DofMapper<N, T> &dof_mapper, const LoadFunctor &boundary_value)
        : DirichletBoundaryVisitor<N, T>(dof_mapper, boundary_value) {}

protected:
    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail,
                             std::vector<int> &bilinear_form_trail_indices,
                             Matrix &bilinear_form_test,
                             std::vector<int> &bilinear_form_test_indices,
                             Matrix &linear_form_value,
                             Matrix &linear_form_test,
                             std::vector<int> &linear_form_test_indices,
                             T &integral_weight,
                             Edge<N, T> *edge,
                             const Quadrature &u) const;
};
template<int N, typename T>
void
BiharmonicDirichletBoundaryVisitor<N, T>::IntegralElementAssembler(Matrix &bilinear_form_trail,
                                                                   std::vector<int> &bilinear_form_trail_indices,
                                                                   Matrix &bilinear_form_test,
                                                                   std::vector<int> &bilinear_form_test_indices,
                                                                   Matrix &linear_form_value,
                                                                   Matrix &linear_form_test,
                                                                   std::vector<int> &linear_form_test_indices,
                                                                   T &integral_weight,
                                                                   Edge<N, T> *edge,
                                                                   const Quadrature &u) const
{
    auto edge_domain = edge->GetDomain();
    auto trial_domain = edge->Parent(0).lock()->GetDomain();
    //    set up integration weights
    integral_weight = u.second * edge_domain->Jacobian(u.first);
    Vector trial_quadrature_abscissa;
    if (!Accessory::MapParametricPoint(&*edge_domain, u.first, &*trial_domain, trial_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    // set up trial basis functions and test basis functions at integration point as matrices
    auto evals = trial_domain->Eval1PhyDerAllTensor(trial_quadrature_abscissa);
    linear_form_value.resize(2, 1);
    linear_form_value(0, 0) = this->_dirichletFunctor(trial_domain->AffineMap(trial_quadrature_abscissa))[0];
    T normal_derivative{0};
    Eigen::Matrix<T, N, 1> normal = edge->NormalDirection(u.first);
    for (int i = 0; i < N; ++i)
    {
        normal_derivative +=
            normal(i) * this->_dirichletFunctor(trial_domain->AffineMap(trial_quadrature_abscissa))[i + 1];
    }
    linear_form_value(1, 0) = normal_derivative;
    linear_form_test.resize(2, evals->size());
    bilinear_form_trail.resize(2, evals->size());
    bilinear_form_test.resize(2, evals->size());
    for (int j = 0; j < evals->size(); ++j)
    {
        linear_form_test(0, j) = (*evals)[j].second[0];
        T normal_derivative{0};
        for (int i = 0; i < N; ++i)
        {
            normal_derivative += normal(i) * (*evals)[j].second[1 + i];
        }
        linear_form_test(1, j) = normal_derivative;
    }
    bilinear_form_trail = linear_form_test;
    bilinear_form_test = bilinear_form_trail;

    // set up indices cooresponding to test basis functions and trial basis functions
    if (bilinear_form_test_indices.size() == 0)
    {
        bilinear_form_test_indices = trial_domain->ActiveIndex(trial_quadrature_abscissa);
        this->_dofMapper.IndicesToGlobal(trial_domain, bilinear_form_test_indices);
    }
    if (bilinear_form_trail_indices.size() == 0)
    {
        bilinear_form_trail_indices = bilinear_form_test_indices;
    }
    if (linear_form_test_indices.size() == 0)
    {
        linear_form_test_indices = bilinear_form_test_indices;
    }
}
