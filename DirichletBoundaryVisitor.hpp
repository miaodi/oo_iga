//
// Created by miaodi on 19/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"

template<int N, typename T>
class DofMapper;

template<int N, typename T>
class DirichletBoundaryVisitor : public DomainVisitor<1, N, T>
{
public:
    using Knot = typename DomainVisitor<1, N, T>::Knot;
    using Quadrature = typename DomainVisitor<1, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<1, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<1, N, T>::KnotSpan;
    using KnotSpanlist  = typename DomainVisitor<1, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<1, N, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<1, N, T>::Matrix;
    using Vector = typename DomainVisitor<1, N, T>::Vector;
public:
    DirichletBoundaryVisitor(const DofMapper<N, T> &dof_mapper, const LoadFunctor &body_force)
        : DomainVisitor<2, N, T>(dof_mapper), _DirichletFunctor(body_force) {}

    void
    DirichletBoundary(Eigen::SparseVector<T> &boundary) const;

protected:
    std::vector<Eigen::Triplet<T>> _gramian;
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor &_DirichletFunctor;
};

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::DirichletBoundary(Eigen::SparseVector<T> &boundary) const
{
    std::vector<Eigen::Triplet<T>> condensed_gramian;
    std::vector<Eigen::Triplet<T>> condensed_rhs;
    auto dirichlet_map = this->_dofMapper.GlobalDirichletCondensedMap();
    auto dirichlet_indices = this->_dofMapper.GlobalDirichletIndices();
    this->CondensedTripletVia(dirichlet_map, dirichlet_map, _gramian, condensed_gramian);
    this->CondensedTripletVia(dirichlet_map, _rhs, condensed_rhs);
    Eigen::SparseMatrix<T> gramian_matrix, rhs_vector;
    this->MatrixAssembler(dirichlet_map.size(), dirichlet_map.size(), _gramian, gramian_matrix);
    this->VectorAssembler(dirichlet_map.size(), _rhs, rhs_vector);
    Vector res = this->Solve(gramian_matrix, rhs_vector);
    boundary.resize(_dofMapper.Dof(), 1);
    for (int i = 0; i < res.rows(); ++i)
    {
        boundary.coeffRef(dirichlet_indices[i]) = res(i);
    }
}
