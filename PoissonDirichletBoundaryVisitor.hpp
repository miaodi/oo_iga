//
// Created by miaodi on 22/10/2017.
//

# pragma once

#include "DirichletBoundaryVisitor.hpp"
#include "DofMapper.hpp"

template<int N, typename T>
class PoissonDirichletBoundaryVisitor : public DirichletBoundaryVisitor<N, T> {
    using Knot = typename DirichletBoundaryVisitor<N, T>::Knot;
    using Quadrature = typename DirichletBoundaryVisitor<N, T>::Quadrature;
    using QuadList = typename DirichletBoundaryVisitor<N, T>::QuadList;
    using KnotSpan = typename DirichletBoundaryVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename DirichletBoundaryVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename DirichletBoundaryVisitor<N, T>::LoadFunctor;
    using Matrix =  typename DirichletBoundaryVisitor<N, T>::Matrix;
    using Vector = typename DirichletBoundaryVisitor<N, T>::Vector;
    using DomainShared_ptr = typename DirichletBoundaryVisitor<N, T>::DomainShared_ptr;
public:
    PoissonDirichletBoundaryVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor& boundary_value)
            :DirichletBoundaryVisitor<N, T>(dof_mapper, boundary_value) { }

protected:
    virtual void
    IntegralElementAssembler(Matrix& bilinear_form_trail, Matrix& bilinear_form_test, Matrix& linear_form_value,
            Matrix& linear_form_test, const DomainShared_ptr domain, const Knot& u) const;
};

template<int N, typename T>
void PoissonDirichletBoundaryVisitor<N, T>::IntegralElementAssembler(
        PoissonDirichletBoundaryVisitor::Matrix& bilinear_form_trail,
        PoissonDirichletBoundaryVisitor::Matrix& bilinear_form_test,
        PoissonDirichletBoundaryVisitor::Matrix& linear_form_value,
        PoissonDirichletBoundaryVisitor::Matrix& linear_form_test,
        const PoissonDirichletBoundaryVisitor::DomainShared_ptr domain,
        const PoissonDirichletBoundaryVisitor::Knot& u) const
{
    auto evals = domain->EvalDerAllTensor(u, 0);
    linear_form_value.resize(1, 1);
    linear_form_value(0, 0) = this->_dirichletFunctor(domain->AffineMap(u))[0];
    linear_form_test.resize(1, evals->size());
    bilinear_form_trail.resize(1, evals->size());
    for (int j = 0; j<evals->size(); ++j)
    {
        linear_form_test(0, j) = (*evals)[j].second[0];
    }
    bilinear_form_trail = linear_form_test;
    bilinear_form_test = bilinear_form_trail;
}
