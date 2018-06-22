//
// Created by miaodi on 21/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"

template <int N, int TrialFunctionDimension, typename T, int d = 2>
class StiffnessVisitor : public DomainVisitor<d, N, T>
{
  public:
    using Knot = typename DomainVisitor<d, N, T>::Knot;
    using Quadrature = typename DomainVisitor<d, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<d, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<d, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<d, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<d, N, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<d, N, T>::Matrix;
    using Vector = typename DomainVisitor<d, N, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<d, N, T>>;

  public:
    StiffnessVisitor(const LoadFunctor &body_force) : _bodyForceFunctor(body_force) {}

    void
    StiffnessAssembler(Eigen::SparseMatrix<T> &) const;

    void
    LoadAssembler(Eigen::SparseMatrix<T> &) const;

    int ID() const
    {
        return _ID;
    }

    const std::vector<Eigen::Triplet<T>> &GetStiffness() const
    {
        return _stiffnees;
    }

    const std::vector<Eigen::Triplet<T>> &GetRhs() const
    {
        return _rhs;
    }

  protected:
    //    Assemble stiffness matrix and rhs
    void
        LocalAssemble(Element<d, N, T> *, const QuadratureRule<T> &, const KnotSpan &);

    virtual void
    IntegralElementAssembler(Matrix &bilinear_form_trail, Matrix &bilinear_form_test, Matrix &linear_form_value,
                             Matrix &linear_form_test, const DomainShared_ptr domain, const Knot &u) const = 0;

    void Initialize(Element<d, N, T> *g)
    {
        _ID = g->GetID();
    }

  protected:
    std::vector<Eigen::Triplet<T>> _stiffnees;
    std::vector<Eigen::Triplet<T>> _rhs;
    int _ID;
    const LoadFunctor &_bodyForceFunctor;
};

template <int N, int TrialFunctionDimension, typename T, int d>
void
    StiffnessVisitor<N, TrialFunctionDimension, T, d>::LocalAssemble(Element<d, N, T> *g,
                                                                  const QuadratureRule<T> &quadrature_rule,
                                                                  const KnotSpan &knot_span)
{
    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto index = domain->ActiveIndex(quadrature_points[0].first);
    std::vector<int> vector_field_index;
    for (const auto &i : index)
    {
        for (int j = 0; j < TrialFunctionDimension; j++)
        {
            vector_field_index.push_back(TrialFunctionDimension * i + j);
        }
    }
    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> bilinear_form_test_indices{vector_field_index}, bilinear_form_trial_indices{vector_field_index}, linear_form_test_indices{vector_field_index};
    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(num_of_quadrature), linear_form_test(num_of_quadrature), linear_form_value(num_of_quadrature);
    std::vector<T> weights;
    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        weights.push_back(quadrature_points[i].second * domain->Jacobian(quadrature_points[i].first));
        IntegralElementAssembler(bilinear_form_trial[i], bilinear_form_test[i], linear_form_value[i],
                                 linear_form_test[i], domain, quadrature_points[i].first);
    }

    auto stiff = this->LocalStiffness(bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial,
                                      bilinear_form_trial_indices, weights);
    auto load = this->LocalRhs(linear_form_test, linear_form_test_indices, linear_form_value, weights);
    std::lock_guard<std::mutex> lock(this->_mutex);
    this->Triplet(load, _rhs);
    this->SymmetricTriplet(stiff, _stiffnees);
}

template <int N, int TrialFunctionDimension, typename T, int d>
void StiffnessVisitor<N, TrialFunctionDimension, T, d>::StiffnessAssembler(Eigen::SparseMatrix<T> &sparse_matrix) const
{
    auto col_set = Accessory::ColIndicesSet(_stiffnees);
    auto row_set = Accessory::RowIndicesSet(_stiffnees);
    this->MatrixAssembler(*row_set.rbegin() + 1, *col_set.rbegin() + 1, _stiffnees, sparse_matrix);
}

template <int N, int TrialFunctionDimension, typename T, int d>
void StiffnessVisitor<N, TrialFunctionDimension, T, d>::LoadAssembler(Eigen::SparseMatrix<T> &sparse_matrix) const
{
    auto row_set = Accessory::RowIndicesSet(_rhs);
    this->VectorAssembler(*row_set.rbegin() + 1, _rhs, sparse_matrix);
}
