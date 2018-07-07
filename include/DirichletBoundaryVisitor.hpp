//
// Created by miaodi on 19/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"
#include "Edge.hpp"
#include "Utility.hpp"

template <int N, typename T>
class DirichletBoundaryVisitor : public DomainVisitor<1, N, T>
{
public:
    using Knot = typename DomainVisitor<1, N, T>::Knot;
    using Quadrature = typename DomainVisitor<1, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<1, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<1, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<1, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<1, N, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<1, N, T>::Matrix;
    using Vector = typename DomainVisitor<1, N, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

public:
    DirichletBoundaryVisitor( const LoadFunctor& boundary_value ) : _dirichletFunctor( boundary_value )
    {
    }
    bool SolveDirichletBoundary() const;

    auto DirichletBoundaryValue() const
    {
        return _dirichlet;
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           std::vector<int>& bilinear_form_trail_indices,
                                           Matrix& bilinear_form_test,
                                           std::vector<int>& bilinear_form_test_indices,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           std::vector<int>& linear_form_test_indices,
                                           T& integral_weight,
                                           Edge<N, T>* edge,
                                           const Quadrature& u ) const = 0;

    void LocalAssemble( Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan& );

protected:
    std::vector<Eigen::Triplet<T>> _gramian;
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor& _dirichletFunctor;
    mutable std::vector<std::pair<int, T>> _dirichlet;
};

template <int N, typename T>
bool DirichletBoundaryVisitor<N, T>::SolveDirichletBoundary() const
{
    if(_gramian.size()==0){
        return false;
    }
    std::vector<Eigen::Triplet<T>> condensed_gramian;
    std::vector<Eigen::Triplet<T>> condensed_rhs;
    std::vector<int> dirichlet_indices = Accessory::ColIndicesVector( _gramian );
    auto dirichlet_inverse_map = Accessory::IndicesInverseMap( dirichlet_indices );
    this->CondensedTripletVia( dirichlet_inverse_map, dirichlet_inverse_map, _gramian, condensed_gramian );
    this->CondensedTripletVia( dirichlet_inverse_map, _rhs, condensed_rhs );
    Eigen::SparseMatrix<T> gramian_matrix_triangle, rhs_vector, gramian_matrix;
    this->MatrixAssembler( dirichlet_inverse_map.size(), dirichlet_inverse_map.size(), condensed_gramian, gramian_matrix_triangle );
    this->VectorAssembler( dirichlet_inverse_map.size(), condensed_rhs, rhs_vector );
    gramian_matrix = gramian_matrix_triangle.template selfadjointView<Eigen::Upper>();
    Vector res = this->SolveLU( gramian_matrix, rhs_vector );
    for ( int i = 0; i < res.rows(); ++i )
    {
        _dirichlet.push_back( std::make_pair( dirichlet_indices[i], res( i ) ) );
    }
    return true;
}

template <int N, typename T>
void DirichletBoundaryVisitor<N, T>::LocalAssemble( Element<1, N, T>* g, const QuadratureRule<T>& quadrature_rule, const DirichletBoundaryVisitor<N, T>::KnotSpan& knot_span )
{
    auto edge = dynamic_cast<Edge<N, T>*>( g );
    QuadList edge_quadrature_points;
    quadrature_rule.MapToQuadrature( knot_span, edge_quadrature_points );
    auto num_of_quadrature = edge_quadrature_points.size();
    std::vector<Matrix> bilinear_form_test( num_of_quadrature ), bilinear_form_trial( num_of_quadrature ), linear_form_test( num_of_quadrature ),
        linear_form_value( num_of_quadrature );
    std::vector<int> bilinear_form_test_indices, bilinear_form_trial_indices, linear_form_test_indices;
    std::vector<T> weights( num_of_quadrature );
    for ( int i = 0; i < num_of_quadrature; ++i )
    {
        IntegralElementAssembler( bilinear_form_trial[i], bilinear_form_trial_indices, bilinear_form_test[i], bilinear_form_test_indices, linear_form_value[i],
                                  linear_form_test[i], linear_form_test_indices, weights[i], edge, edge_quadrature_points[i] );
    }

    auto stiff = this->LocalStiffness( bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial, bilinear_form_trial_indices, weights );
    auto load = this->LocalRhs( linear_form_test, linear_form_test_indices, linear_form_value, weights );
    std::lock_guard<std::mutex> lock( this->_mutex );
    this->SymmetricTriplet( stiff, _gramian );
    this->Triplet( load, _rhs );
}
