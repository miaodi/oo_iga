//
// Created by miaodi on 30/11/2017.
//

#pragma once

#include "DofMapper.hpp"
#include "StiffnessVisitor.hpp"

template <typename T>
class 2DElasticityDeviatoricStiffnessVisitor : public StiffnessVisitor<2, 2, T>
{
  public:
    using Knot = typename StiffnessVisitor<2, 2, T>::Knot;
    using Quadrature = typename StiffnessVisitor<2, 2, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<2, 2, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<2, 2, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<2, 2, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<2, 2, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<2, 2, T>::Matrix;
    using Vector = typename StiffnessVisitor<2, 2, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<2, 2, T>::DomainShared_ptr;

  public:
    DeviatoricStiffnessVisitor(const LoadFunctor &body_force)
        : StiffnessVisitor<N, T>(body_force)
    {
        T nu = .3;
        T E = 1e11 _constitutive.resize(3, 3);
        _constitutive << 1 - nu, nu, 0, nu, 1 - nu, 0, 0, 0, (1.0 - 2 * nu) / 2;
        _constitutive *= E / (1 + nu) / (1 - 2 * nu);
    }

  protected:
    virtual void
    IntegralElementAssembler(Matrix & bilinear_form_trail, Matrix & bilinear_form_test, Matrix & linear_form_value,
                             Matrix & linear_form_test, const DomainShared_ptr domain, const Knot &u) const;
    Matrix _constitutive;
};

template <int N, typename T>
    void 2DElasticityDeviatoricStiffnessVisitor < N,
    T > ::IntegralElementAssembler(
            2DElasticityDeviatoricStiffnessVisitor < N, T > ::Matrix & bilinear_form_trail,
            2DElasticityDeviatoricStiffnessVisitor < N, T > ::Matrix & bilinear_form_test,
            2DElasticityDeviatoricStiffnessVisitor < N, T > ::Matrix & linear_form_value,
            2DElasticityDeviatoricStiffnessVisitor < N, T > ::Matrix & linear_form_test,
            const 2DElasticityDeviatoricStiffnessVisitor < N, T > ::DomainShared_ptr domain,
            const 2DElasticityDeviatoricStiffnessVisitor < N, T > ::Knot & u) const
{
    auto evals = domain->Eval1PhyDerAllTensor(u);
    linear_form_value.resize(2, 1);
    linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
    linear_form_value(1, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[1];
    linear_form_test.resize(2, 2 * evals->size());
    linear_form_test.setZero();
    bilinear_form_trail.resize(3, 2 * evals->size());
    bilinear_form_trail.setZero();
    for (int j = 0; j < evals->size(); ++j)
    {
        linear_form_test(0, 2 * j) = (*evals)[j].second[0];
        linear_form_test(1, 2 * j + 1) = (*evals)[j].second[0];
        bilinear_form_trail(0, 2 * j) = (*evals)[j].second[1];
        bilinear_form_trail(1, 2 * j + 1) = (*evals)[j].second[2];
        bilinear_form_trail(2, 2 * j) = (*evals)[j].second[2];
        bilinear_form_trail(2, 2 * j + 1) = (*evals)[j].second[1];
    }
    bilinear_form_test = _constitutive * bilinear_form_trail;
}
