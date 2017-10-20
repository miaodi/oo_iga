//
// Created by miaodi on 19/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"

template<int N, typename T>
class DofMapper;

template<int N, typename T>
class DirichletBoundaryVisitor : public DomainVisitor<1, N, T> {
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
    DirichletBoundaryVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor& body_force)
            :DomainVisitor<2, N, T>(dof_mapper), _DirichletFunctor(body_force) { }

    void

    void DirichletBoundary(Eigen::Triplet<T>& boundary) const;

protected:
    std::vector<MatrixData<T>> _Gramian;
    std::vector<VectorData<T>> _rhs;
    const LoadFunctor& _DirichletFunctor;
};

template<int N, typename T>
void DirichletBoundaryVisitor<N, T>::DirichletBoundary(Eigen::Triplet& boundary) const
{

}
