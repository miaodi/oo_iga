//
// Created by miaodi on 30/11/2017.
//

#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class MembraneStiffnessVisitor : public StiffnessVisitor<3, 3, T>
{
  public:
    using Knot = typename StiffnessVisitor<3, 3, T>::Knot;
    using Quadrature = typename StiffnessVisitor<3, 3, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<3, 3, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<3, 3, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<3, 3, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<3, 3, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<3, 3, T>::Matrix;
    using Vector = typename StiffnessVisitor<3, 3, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<3, 3, T>::DomainShared_ptr;

  public:
    MembraneStiffnessVisitor(const LoadFunctor &body_force)
        : StiffnessVisitor<3, 3, T>(body_force) {}

  protected:
    virtual void IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                                          Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const;

  protected:
    T _nu{.0};
    T _E{4.32e8};
    T _h{0.25};
};

template <typename T>
void MembraneStiffnessVisitor<T>::IntegralElementAssembler(Matrix &bilinear_form_trail,
                                                           Matrix &bilinear_form_test,
                                                           Matrix &linear_form_value,
                                                           Matrix &linear_form_test,
                                                           const DomainShared_ptr domain,
                                                           const Knot &u) const
{
    auto evals = domain->EvalDerAllTensor(u, 1);
    linear_form_value.resize(3, 1);
    linear_form_value(0, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[0];
    linear_form_value(1, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[1];
    linear_form_value(2, 0) = this->_bodyForceFunctor(domain->AffineMap(u))[2];

    linear_form_test.resize(3, 3 * evals->size());
    linear_form_test.setZero();
    bilinear_form_trail.resize(3, 3 * evals->size());
    bilinear_form_trail.setZero();

    Eigen::Matrix<T, 3, 1> u1, u2, u3, v1, v2, v3;
    u1 = domain->AffineMap(u, {1, 0});
    u2 = domain->AffineMap(u, {0, 1});
    u3 = u1.cross(u2);
    T jacobian = u3.norm();
    u3 *= 1.0 / jacobian;

    std::tie(v1, v2, v3) = Accessory::CovariantToContravariant(u1, u2, u3);

    for (int j = 0; j < evals->size(); ++j)
    {
        linear_form_test(0, 3 * j) = (*evals)[j].second[0];
        linear_form_test(1, 3 * j + 1) = (*evals)[j].second[0];
        linear_form_test(2, 3 * j + 2) = (*evals)[j].second[0];

        bilinear_form_trail(0, 3 * j) = (*evals)[j].second[1] * u1(0);
        bilinear_form_trail(0, 3 * j + 1) = (*evals)[j].second[1] * u1(1);
        bilinear_form_trail(0, 3 * j + 2) = (*evals)[j].second[1] * u1(2);

        bilinear_form_trail(1, 3 * j) = (*evals)[j].second[2] * u2(0);
        bilinear_form_trail(1, 3 * j + 1) = (*evals)[j].second[2] * u2(1);
        bilinear_form_trail(1, 3 * j + 2) = (*evals)[j].second[2] * u2(2);

        bilinear_form_trail(2, 3 * j) = (*evals)[j].second[1] * u2(0) + (*evals)[j].second[2] * u1(0);
        bilinear_form_trail(2, 3 * j + 1) = (*evals)[j].second[1] * u2(1) + (*evals)[j].second[2] * u1(1);
        bilinear_form_trail(2, 3 * j + 2) = (*evals)[j].second[1] * u2(2) + (*evals)[j].second[2] * u1(2);
    }

    T v11, v12, v22;
    v11 = v1.dot(v1);
    v22 = v2.dot(v2);
    v12 = v1.dot(v2);

    Matrix H(3, 3);
    H << v11 * v11, _nu * v11 * v22 + (1 - _nu) * v12 * v12, v11 * v12, _nu * v11 * v22 + (1 - _nu) * v12 * v12, v22 * v22, v22 * v12, v11 * v12, v22 * v12, .5 * ((1 - _nu) * v11 * v22 + (1 + _nu) * v12 * v12);
    bilinear_form_test = _E * _h / (1 - _nu * _nu) * H * bilinear_form_trail;
}
