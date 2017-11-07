//
// Created by miaodi on 21/10/2017.
//

#pragma once

#include "DofMapper.hpp"
#include "StiffnessVisitor.hpp"

template<int N, typename T>
class BiharmonicStiffnessVisitor : public StiffnessVisitor<N, T> {
public:
    using Knot = typename StiffnessVisitor<N, T>::Knot;
    using Quadrature = typename StiffnessVisitor<N, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<N, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<N, T>::LoadFunctor;
    using Matrix =  typename StiffnessVisitor<N, T>::Matrix;
    using Vector = typename StiffnessVisitor<N, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<N, T>::DomainShared_ptr;
public:
    BiharmonicStiffnessVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor& body_force)
            :StiffnessVisitor<N, T>(dof_mapper, body_force) { }

protected:
    virtual void
    IntegralElementAssembler(Matrix& bilinear_form_trail, Matrix& bilinear_form_test, Matrix& linear_form_value,
            Matrix& linear_form_test, const DomainShared_ptr domain, const Knot& u) const;


};

template<int N, typename T>
void BiharmonicStiffnessVisitor<N, T>::IntegralElementAssembler(
        BiharmonicStiffnessVisitor<N, T>::Matrix& bilinear_form_trail,
        BiharmonicStiffnessVisitor<N, T>::Matrix& bilinear_form_test,
        BiharmonicStiffnessVisitor<N, T>::Matrix& linear_form_value,
        BiharmonicStiffnessVisitor<N, T>::Matrix& linear_form_test,
        const BiharmonicStiffnessVisitor<N, T>::DomainShared_ptr domain,
        const BiharmonicStiffnessVisitor<N, T>::Knot& u) const
{
    auto evals = domain->Eval2PhyDerAllTensor(u);
    linear_form_value.resize(1, 1);
    linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
    linear_form_test.resize(1, evals->size());
    bilinear_form_trail.resize(1, evals->size());
    for (int j = 0; j<evals->size(); ++j)
    {
        linear_form_test(0, j) = (*evals)[j].second[0];
        bilinear_form_trail(0, j) = (*evals)[j].second[3]+(*evals)[j].second[5];
    }
    bilinear_form_test = bilinear_form_trail;
}
