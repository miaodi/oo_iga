#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class BiharmonicStiffnessVisitor : public StiffnessVisitor<2, 1, T>
{
  public:
    using Knot = typename StiffnessVisitor<2, 1, T>::Knot;
    using Quadrature = typename StiffnessVisitor<2, 1, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<2, 1, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<2, 1, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<2, 1, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<2, 1, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<2, 1, T>::Matrix;
    using Vector = typename StiffnessVisitor<2, 1, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<2, 1, T>::DomainShared_ptr;
    using DataType = T;

  public:
    BiharmonicStiffnessVisitor(const LoadFunctor &body_force)
        : StiffnessVisitor<2, 1, T>(body_force) {}

  protected:
    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                             Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const;
};

template <typename T>
void BiharmonicStiffnessVisitor<T>::IntegralElementAssembler(
    BiharmonicStiffnessVisitor<T>::Matrix &bilinear_form_trail,
    BiharmonicStiffnessVisitor<T>::Matrix &bilinear_form_test,
    BiharmonicStiffnessVisitor<T>::Matrix &linear_form_value,
    BiharmonicStiffnessVisitor<T>::Matrix &linear_form_test,
    const BiharmonicStiffnessVisitor<T>::DomainShared_ptr domain,
    const BiharmonicStiffnessVisitor<T>::Knot &u) const
{
    auto evals = domain->Eval2PhyDerAllTensor(u);
    linear_form_value.resize(1, 1);
    linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
    linear_form_test.resize(1, evals->size());
    bilinear_form_trail.resize(1, evals->size());
    for (int j = 0; j < evals->size(); ++j)
    {
        linear_form_test(0, j) = (*evals)[j].second[0];
        bilinear_form_trail(0, j) = (*evals)[j].second[3] + (*evals)[j].second[5];
    }
    bilinear_form_test = bilinear_form_trail;
}