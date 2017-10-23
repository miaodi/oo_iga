//
// Created by miaodi on 21/10/2017.
//

#pragma once

#include "DofMapper.hpp"
#include "DomainVisitor.hpp"

template<int N, typename T>
class StiffnessVisitor : public DomainVisitor<2, N, T> {
public:
    using Knot = typename DomainVisitor<2, N, T>::Knot;
    using Quadrature = typename DomainVisitor<2, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<2, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<2, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<2, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<2, N, T>::LoadFunctor;
    using Matrix =  typename DomainVisitor<2, N, T>::Matrix;
    using Vector = typename DomainVisitor<2, N, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;
public:
    StiffnessVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor& body_force)
            :DomainVisitor<2, N, T>(dof_mapper), _bodyForceFunctor(body_force) { }

    void
    StiffnessAssembler(Eigen::SparseMatrix<T>&) const;

    void
    LoadAssembler(Eigen::SparseMatrix<T>&) const;

protected:
    //    Assemble stiffness matrix and rhs
    void
    LocalAssemble(Element<2, N, T>*, const QuadratureRule<T>&, const KnotSpan&, std::mutex&);

    virtual void
    IntegralElementAssembler(Matrix& bilinear_form_trail, Matrix& bilinear_form_test, Matrix& linear_form_value,
            Matrix& linear_form_test, const DomainShared_ptr domain, const Knot& u) const = 0;

protected:

    std::vector<Eigen::Triplet<T>> _stiffnees;
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor& _bodyForceFunctor;
};

template<int N, typename T>
void
StiffnessVisitor<N, T>::LocalAssemble(Element<2, N, T>* g,
        const QuadratureRule<T>& quadrature_rule,
        const StiffnessVisitor<N, T>::KnotSpan& knot_span,
        std::mutex& pmutex)
{
    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto index = domain->ActiveIndex(quadrature_points[0].first);
    this->_dofMapper.IndicesToGlobal(domain, index);
    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> bilinear_form_test_indices{index}, bilinear_form_trial_indices{index}, linear_form_test_indices{
            index};
    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(
            num_of_quadrature), linear_form_test(num_of_quadrature), linear_form_value(num_of_quadrature);
    std::vector<T> weights;
    for (int i = 0; i<quadrature_points.size(); ++i)
    {
        weights.push_back(quadrature_points[i].second*domain->Jacobian(quadrature_points[i].first));
        IntegralElementAssembler(bilinear_form_trial[i], bilinear_form_test[i], linear_form_value[i],
                linear_form_test[i], domain, quadrature_points[i].first);
    }

    auto stiff = this->LocalStiffness(bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial,
            bilinear_form_trial_indices, weights);
    auto load = this->LocalRhs(linear_form_test, linear_form_test_indices, linear_form_value, weights);
    std::lock_guard<std::mutex> lock(pmutex);
    this->SymmetricTriplet(stiff, _stiffnees);
    this->Triplet(load, _rhs);
}

template<int N, typename T>
void
StiffnessVisitor<N, T>::StiffnessAssembler(Eigen::SparseMatrix<T>& sparse_matrix) const
{
    this->MatrixAssembler(this->_dofMapper.Dof(), this->_dofMapper.Dof(), _stiffnees, sparse_matrix);
}

template<int N, typename T>
void
StiffnessVisitor<N, T>::LoadAssembler(Eigen::SparseMatrix<T>& sparse_matrix) const
{
    this->VectorAssembler(this->_dofMapper.Dof(), _rhs, sparse_matrix);
}
