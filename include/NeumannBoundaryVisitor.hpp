//
// Created by miaodi on 21/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"

template <typename T>
class NeumannBoundaryVisitor : public DomainVisitor<1, 2, T>
{
public:
    using Knot = typename DomainVisitor<1, 2, T>::Knot;
    using Quadrature = typename DomainVisitor<1, 2, T>::Quadrature;
    using QuadList = typename DomainVisitor<1, 2, T>::QuadList;
    using KnotSpan = typename DomainVisitor<1, 2, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<1, 2, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<1, 2, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<1, 2, T>::Matrix;
    using Vector = typename DomainVisitor<1, 2, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<1, 2, T>>;

public:
    NeumannBoundaryVisitor( const LoadFunctor& traction ) : _tractionFunctor( traction )
    {
    }

    void NeumannBoundaryAssembler( Eigen::SparseMatrix<T>& ) const;

protected:
    //    Assemble stiffness matrix and rhs
    void LocalAssemble( Element<1, 2, T>*, const QuadratureRule<T>&, const KnotSpan& );

    virtual void IntegralElementAssembler(
        Matrix& linear_form_value, std::vector<int>&, Matrix& linear_form_test, const Edge<2, T>* edge, const Knot& u ) const;

protected:
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor& _tractionFunctor;
    int _size;
};

template <typename T>
void NeumannBoundaryVisitor<T>::LocalAssemble( Element<1, 2, T>* g, const QuadratureRule<T>& quadrature_rule, const KnotSpan& knot_span )
{
    auto edge = dynamic_cast<Edge<2, T>*>( g );
    auto domain = g->GetDomain();
    auto surface_domain = edge->Parent( 0 ).lock()->GetDomain();
    _size = 2 * surface_domain->GetDof();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature( knot_span, quadrature_points );

    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> linear_form_test_indices;
    std::vector<Matrix> linear_form_test( num_of_quadrature ), linear_form_value( num_of_quadrature );
    std::vector<T> weights;

    for ( int i = 0; i < quadrature_points.size(); ++i )
    {
        weights.push_back( quadrature_points[i].second * domain->Jacobian( quadrature_points[i].first ) );
        IntegralElementAssembler( linear_form_value[i], linear_form_test_indices, linear_form_test[i], edge,
                                  quadrature_points[i].first );
    }

    auto load = this->LocalRhs( linear_form_test, linear_form_test_indices, linear_form_value, weights );
    std::lock_guard<std::mutex> lock( this->_mutex );
    this->Triplet( load, _rhs );
}
template <typename T>
void NeumannBoundaryVisitor<T>::NeumannBoundaryAssembler( Eigen::SparseMatrix<T>& sparse_matrix ) const
{
    auto row_set = Accessory::RowIndicesSet( _rhs );
    this->VectorAssembler( _size, _rhs, sparse_matrix );
}

template <typename T>
void NeumannBoundaryVisitor<T>::IntegralElementAssembler( Matrix& linear_form_value,
                                                          std::vector<int>& linear_form_test_indices,
                                                          Matrix& linear_form_test,
                                                          const Edge<2, T>* edge,
                                                          const Knot& u ) const
{
    auto domain = edge->GetDomain();
    auto surface_domain = edge->Parent( 0 ).lock()->GetDomain();

    // TODO: bug for inverse mapping using nurbs
    Vector trial_quadrature_abscissa( 2 );
    if ( !Accessory::MapParametricPoint( &*domain, u, &*surface_domain, trial_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    auto test_evals = surface_domain->EvalDerAllTensor( trial_quadrature_abscissa );
    linear_form_value.resize( 2, 1 );
    linear_form_test.resize( 2, 2 * test_evals->size() );
    linear_form_test.setZero();

    T sigma_xx = _tractionFunctor( surface_domain->AffineMap( trial_quadrature_abscissa ) )[0];
    T sigma_yy = _tractionFunctor( surface_domain->AffineMap( trial_quadrature_abscissa ) )[1];
    T sigma_xy = _tractionFunctor( surface_domain->AffineMap( trial_quadrature_abscissa ) )[2];
    Matrix stress_tensor( 2, 2 );
    stress_tensor << sigma_xx, sigma_xy, sigma_xy, sigma_yy;

    // TODO: Bug for create edge domain for NURBS
    Eigen::Matrix<T, 2, 1> normal = edge->NormalDirection( u );

    linear_form_value = stress_tensor * normal;
    for ( int j = 0; j < test_evals->size(); ++j )
    {
        linear_form_test( 0, 2 * j ) = ( *test_evals )[j].second[0];
        linear_form_test( 1, 2 * j + 1 ) = ( *test_evals )[j].second[0];
    }
    if ( linear_form_test_indices.size() == 0 )
    {
        auto indices = surface_domain->ActiveIndex( trial_quadrature_abscissa );
        for ( auto i : indices )
        {
            linear_form_test_indices.push_back( 2 * i );
            linear_form_test_indices.push_back( 2 * i + 1 );
        }
    }
}
