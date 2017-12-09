//
// Created by miaodi on 12/03/2017.
//

#pragma once

#include "DomainVisitor.hpp"

template <typename T>
class PressureProjectionVisitor : public DomainVisitor<2, 2, T>
{
  public:
    using Knot = typename DomainVisitor<2, 2, T>::Knot;
    using Quadrature = typename DomainVisitor<2, 2, T>::Quadrature;
    using QuadList = typename DomainVisitor<2, 2, T>::QuadList;
    using KnotSpan = typename DomainVisitor<2, 2, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<2, 2, T>::KnotSpanlist;
    using Matrix = typename DomainVisitor<2, 2, T>::Matrix;
    using Vector = typename DomainVisitor<2, 2, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, 2, T>>;

  public:
    PressureProjectionVisitor(const bool &bezier) : _bezierDual(bezier) {}

    void
    InnerProductAssembler(Eigen::SparseMatrix<T> &) const;

  protected:
    void Initialize(Element<2, 2, T> *g);

    //    Assemble stiffness matrix and rhs
    void LocalAssemble(Element<2, 2, T> *, const QuadratureRule<T> &, const KnotSpan &);

    virtual void
    IntegralElementAssembler(Matrix &left_term, Matrix &right_term, const DomainShared_ptr domain, const Knot &u) const;

  protected:
    std::vector<Eigen::Triplet<T>> _innerProduct;
    TensorBsplineBasis<2, T> _pressureParametricDomain;
    bool _bezierDual;
};

template <typename T>
void PressureProjectionVisitor<T>::Initialize(Element<2, 2, T> *g)
{

    auto knot_x = g->GetDomain()->KnotVectorGetter(0);
    auto knot_y = g->GetDomain()->KnotVectorGetter(1);
    knot_x.erase(knot_x.begin());
    knot_x.erase(knot_x.end() - 1);
    knot_y.erase(knot_y.begin());
    knot_y.erase(knot_y.end() - 1);
    _pressureParametricDomain.KnotVectorSetter(knot_x, 0);
    _pressureParametricDomain.KnotVectorSetter(knot_y, 1);
    _pressureParametricDomain.BezierDualInitialize();
}

template <typename T>
void PressureProjectionVisitor<T>::LocalAssemble(Element<2, 2, T> *g,
                                                 const QuadratureRule<T> &quadrature_rule,
                                                 const KnotSpan &knot_span)
{

    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto index = domain->ActiveIndex(quadrature_points[0].first);
    auto test_index = _pressureParametricDomain.ActiveIndex(quadrature_points[0].first);
    std::vector<int> vector_field_index;
    for (const auto &i : index)
    {
        for (int j = 0; j < 2; j++)
        {
            vector_field_index.push_back(2 * i + j);
        }
    }
    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> bilinear_form_test_indices{test_index}, bilinear_form_trial_indices{vector_field_index};
    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(num_of_quadrature);
    std::vector<T> weights;

    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        if (_bezierDual)
        {
            weights.push_back(quadrature_points[i].second);
        }
        else
        {
            weights.push_back(quadrature_points[i].second * domain->Jacobian(quadrature_points[i].first));
        }
        IntegralElementAssembler(bilinear_form_trial[i], bilinear_form_test[i], domain, quadrature_points[i].first);
    }
    auto stiff = this->LocalStiffness(bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial,
                                      bilinear_form_trial_indices, weights);
    std::lock_guard<std::mutex> lock(this->_mutex);
    this->Triplet(stiff, _innerProduct);
}

template <typename T>
void PressureProjectionVisitor<T>::InnerProductAssembler(Eigen::SparseMatrix<T> &sparse_matrix) const
{
    auto col_set = Accessory::ColIndicesSet(_innerProduct);
    auto row_set = Accessory::RowIndicesSet(_innerProduct);
    this->MatrixAssembler(*row_set.rbegin() + 1, *col_set.rbegin() + 1, _innerProduct, sparse_matrix);
}

template <typename T>
void PressureProjectionVisitor<T>::IntegralElementAssembler(
    Matrix &bilinear_form_trail,
    Matrix &bilinear_form_test,
    const DomainShared_ptr domain,
    const Knot &u) const
{

    if (_bezierDual)
    {
        auto test_evals = _pressureParametricDomain.EvalDualAllTensor(u);
        auto evals = domain->Eval1PhyDerAllTensor(u);
        bilinear_form_trail.resize(1, 2 * evals->size());
        bilinear_form_trail.setZero();
        bilinear_form_test.resize(1, test_evals->size());
        bilinear_form_test.setZero();
        for (int j = 0; j < evals->size(); ++j)
        {
            bilinear_form_trail(0, 2 * j) = (*evals)[j].second[1];
            bilinear_form_trail(0, 2 * j + 1) = (*evals)[j].second[2];
        }
        for (int j = 0; j < test_evals->size(); ++j)
        {
            bilinear_form_test(0, j) = (*test_evals)[j].second[0];
        }
    }
    else
    {
        auto test_evals = _pressureParametricDomain.EvalDerAllTensor(u);
        auto evals = domain->Eval1PhyDerAllTensor(u);
        bilinear_form_trail.resize(1, 2 * evals->size());
        bilinear_form_trail.setZero();
        bilinear_form_test.resize(1, test_evals->size());
        bilinear_form_test.setZero();
        for (int j = 0; j < evals->size(); ++j)
        {
            bilinear_form_trail(0, 2 * j) = (*evals)[j].second[1];
            bilinear_form_trail(0, 2 * j + 1) = (*evals)[j].second[2];
        }
        for (int j = 0; j < test_evals->size(); ++j)
        {
            bilinear_form_test(0, j) = (*test_evals)[j].second[0];
        }
    }
}
