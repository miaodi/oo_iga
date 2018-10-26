#pragma once

#include "DofMapper.hpp"
#include "StiffnessVisitor.hpp"


template <int N, typename T>
class H2StiffnessVisitor : public StiffnessVisitor<N, T>
{
  public:
    using Knot = typename StiffnessVisitor<N, T>::Knot;
    using Quadrature = typename StiffnessVisitor<N, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<N, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<N, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<N, T>::Matrix;
    using Vector = typename StiffnessVisitor<N, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<N, T>::DomainShared_ptr;

  public:
    H2StiffnessVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor &body_force)
        : StiffnessVisitor<N, T>(dof_mapper, body_force) {}

  protected:
    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                             Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const override
    {
        auto evals = domain->Eval2PhyDerAllTensor(u);
        linear_form_value.resize(6, 1);
        linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
        linear_form_value(1, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[1];
        linear_form_value(2, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[2];
        linear_form_value(3, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[3];
        linear_form_value(4, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[4];
        linear_form_value(5, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[5];
        linear_form_test.resize(6, evals->size());
        bilinear_form_trail.resize(6, evals->size());
        for (int j = 0; j < evals->size(); ++j)
        {
            linear_form_test(0, j) = (*evals)[j].second[0];
            linear_form_test(1, j) = (*evals)[j].second[1];
            linear_form_test(2, j) = (*evals)[j].second[2];
            linear_form_test(3, j) = (*evals)[j].second[3];
            linear_form_test(4, j) = (*evals)[j].second[4];
            linear_form_test(5, j) = (*evals)[j].second[5];
        }

        bilinear_form_trail = linear_form_test;
        bilinear_form_test = bilinear_form_trail;
    }
};