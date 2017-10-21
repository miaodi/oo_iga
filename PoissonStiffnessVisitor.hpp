//
// Created by di miao on 10/18/17.
//

#pragma once

#include "DofMapper.hpp"
#include "DomainVisitor.hpp"

template<int N, typename T>
class PoissonStiffnessVisitor : public DomainVisitor<2, N, T>
{
public:
    using Knot = typename DomainVisitor<2, N, T>::Knot;
    using Quadrature = typename DomainVisitor<2, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<2, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<2, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<2, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<2, N, T>::LoadFunctor;
    using Matrix =  typename DomainVisitor<2, N, T>::Matrix;
    using Vector = typename DomainVisitor<2, N, T>::Vector;
public:
    PoissonStiffnessVisitor(const DofMapper<N, T> &dof_mapper, const LoadFunctor &body_force)
        : DomainVisitor<2, N, T>(dof_mapper), _bodyForceFunctor(body_force) {}

//    Assemble stiffness matrix and rhs
    void
    LocalAssemble(Element<2, N, T> *, const QuadratureRule<T> &, const KnotSpan &, std::mutex &);

    void
    StiffnessAssembler(Eigen::SparseMatrix<T> &) const;

    void
    LoadAssembler(Eigen::SparseMatrix<T> &) const;

    virtual void Assember()

protected:

    std::vector<Eigen::Triplet<T>> _stiffnees;
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor &_bodyForceFunctor;
};

template<int N, typename T>
void
PoissonStiffnessVisitor<N, T>::LocalAssemble(Element<2, N, T> *g,
                                             const QuadratureRule<T> &quadrature_rule,
                                             const PoissonStiffnessVisitor<N, T>::KnotSpan &knot_span,
                                             std::mutex &pmutex)
{
    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto index = domain->ActiveIndex(quadrature_points[0].first);
    this->_dofMapper.IndicesToGlobal(domain, index);
    auto num_of_basis = index.size();
    auto num_of_quadrature = quadrature_points.size();
    std::vector<int> poisson_weight_indices{index}, poisson_basis_indices{index}, load_weight_indices{index};
    std::vector<Matrix> poisson_weight(num_of_quadrature), poisson_basis(num_of_quadrature),
        load_weight(num_of_quadrature), load_value(num_of_quadrature);
    std::vector<T> weights;
    for (int i = 0; i < quadrature_points.size(); ++i){
        weights.push_back(quadrature_points[i].second * domain->Jacobian(quadrature_points[i].first));
    }

    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        auto evals = domain->Eval1PhyDerAllTensor(quadrature_points[i].first);
        load_value[i].resize(1, 1);
        load_value[i](0, 0) = _bodyForceFunctor(domain->AffineMap(quadrature_points[i].first))[0];
        load_weight[i].resize(1, num_of_basis);
        poisson_basis[i].resize(2, num_of_basis);
        for (int j = 0; j < num_of_basis; ++j)
        {
            load_weight[i](0, j) = (*evals)[j].second[0];
            poisson_basis[i](0, j) = (*evals)[j].second[1];
            poisson_basis[i](1, j) = (*evals)[j].second[2];
        }
    }
    poisson_weight = poisson_basis;

    auto stiff = this->LocalStiffness(poisson_weight, poisson_weight_indices, poisson_basis, poisson_basis_indices,
                                      weights);
    auto load = this->LocalRhs(load_weight, load_weight_indices, load_value, weights);
    std::lock_guard<std::mutex> lock(pmutex);
    this->SymmetricTriplet(stiff, _stiffnees);
    this->Triplet(load, _rhs);
}

template<int N, typename T>
void
PoissonStiffnessVisitor<N, T>::StiffnessAssembler(Eigen::SparseMatrix<T> &sparse_matrix) const
{
    this->MatrixAssembler(this->_dofMapper.Dof(), this->_dofMapper.Dof(), _stiffnees, sparse_matrix);
}

template<int N, typename T>
void
PoissonStiffnessVisitor<N, T>::LoadAssembler(Eigen::SparseMatrix<T> &sparse_matrix) const
{
    this->VectorAssembler(this->_dofMapper.Dof(), _rhs, sparse_matrix);
}
