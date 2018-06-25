#pragma once

#include "StiffnessVisitor.hpp"

template <int d, int N, typename T>
class L2StiffnessVisitor : public StiffnessVisitor<N, 1, T, d>
{
  public:
    using Knot = typename StiffnessVisitor<N, 1, T, d>::Knot;
    using Quadrature = typename StiffnessVisitor<N, 1, T, d>::Quadrature;
    using QuadList = typename StiffnessVisitor<N, 1, T, d>::QuadList;
    using KnotSpan = typename StiffnessVisitor<N, 1, T, d>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<N, 1, T, d>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<N, 1, T, d>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<N, 1, T, d>::Matrix;
    using Vector = typename StiffnessVisitor<N, 1, T, d>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<N, 1, T, d>::DomainShared_ptr;
    using DataType = T;

  public:
    L2StiffnessVisitor(const LoadFunctor &body_force)
        : StiffnessVisitor<N, 1, T, d>(body_force) {}

  protected:
    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                             Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const override
    {
        auto evals = domain->EvalDualAllTensor(u);
        linear_form_value.resize(1, 1);
        linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
        linear_form_test.resize(1, evals->size());
        bilinear_form_trail.resize(1, evals->size());
        for (int j = 0; j < evals->size(); ++j)
        {
            linear_form_test(0, j) = (*evals)[j].second[0];
        }

        bilinear_form_trail = linear_form_test;
        bilinear_form_test = bilinear_form_trail;
    }
};

template <int d, int N, typename T>
class H2StiffnessVisitor : public StiffnessVisitor<N, 1, T, d>
{
  public:
    using Knot = typename StiffnessVisitor<N, 1, T, d>::Knot;
    using Quadrature = typename StiffnessVisitor<N, 1, T, d>::Quadrature;
    using QuadList = typename StiffnessVisitor<N, 1, T, d>::QuadList;
    using KnotSpan = typename StiffnessVisitor<N, 1, T, d>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<N, 1, T, d>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<N, 1, T, d>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<N, 1, T, d>::Matrix;
    using Vector = typename StiffnessVisitor<N, 1, T, d>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<N, 1, T, d>::DomainShared_ptr;
    using DataType = T;

  public:
    H2StiffnessVisitor(const LoadFunctor &body_force)
        : StiffnessVisitor<N, 1, T, d>(body_force) {}

  protected:
    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                             Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const override
    {
        auto evals = domain->Eval2PhyDerAllTensor(u);
        linear_form_value.resize(1, 1);
        linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
        linear_form_test.resize(6, evals->size());
        bilinear_form_trail.resize(6, evals->size());
        for (int j = 0; j < evals->size(); ++j)
        {
            bilinear_form_trail(0, j) = (*evals)[j].second[0];
            bilinear_form_trail(1, j) = (*evals)[j].second[1];
            bilinear_form_trail(2, j) = (*evals)[j].second[2];
            bilinear_form_trail(3, j) = (*evals)[j].second[3];
            bilinear_form_trail(4, j) = (*evals)[j].second[4];
            bilinear_form_trail(5, j) = (*evals)[j].second[5];
        }

        for (int j = 0; j < evals->size(); ++j)
        {
            linear_form_test(0, j) = (*evals)[j].second[0];
        }
        bilinear_form_test = bilinear_form_trail;
    }
};
