//
// Created by miaodi on 21/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"

enum class NeumannBoundaryType
{
    Traction = 0,
    Moment = 1,
};

template <int N, typename T, NeumannBoundaryType nt>
class NeumannBoundaryVisitor : public DomainVisitor<1, N, T>
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
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<1, N, T>>;

public:
    NeumannBoundaryVisitor( const LoadFunctor& traction ) : _tractionFunctor( traction )
    {
    }

    void NeumannBoundaryAssembler( Eigen::SparseMatrix<T>& ) const;

protected:
    //    Assemble stiffness matrix and rhs
    void LocalAssemble( Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan& );

    template <NeumannBoundaryType tt = nt>
    typename std::enable_if<tt == NeumannBoundaryType::Traction && N == 2, void>::type IntegralElementAssembler(
        Matrix& linear_form_value, std::vector<int>& linear_form_test_indices, Matrix& linear_form_test, const Edge<N, T>* edge, const Knot& u ) const
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

    template <NeumannBoundaryType tt = nt>
    typename std::enable_if<tt == NeumannBoundaryType::Traction && N == 3, void>::type IntegralElementAssembler(
        Matrix& linear_form_value, std::vector<int>& linear_form_test_indices, Matrix& linear_form_test, const Edge<N, T>* edge, const Knot& u ) const
    {
        auto domain = edge->GetDomain();
        auto surface_domain = edge->Parent( 0 ).lock()->GetDomain();

        // TODO: bug for inverse mapping using nurbs
        Vector trial_quadrature_abscissa( 2 );
        trial_quadrature_abscissa << 1, u( 0 );
        // if ( !Accessory::MapParametricPoint( &*domain, u, &*surface_domain, trial_quadrature_abscissa ) )
        // {
        //     std::cout << "MapParametericPoint failed" << std::endl;
        // }
        auto test_evals = surface_domain->EvalDerAllTensor( trial_quadrature_abscissa );
        linear_form_value.resize( 3, 1 );
        linear_form_test.resize( 3, 3 * test_evals->size() );
        linear_form_test.setZero();

        linear_form_value( 0 ) = 0;
        linear_form_value( 1 ) = 0;
        linear_form_value( 2 ) = .04 * _tractionFunctor( trial_quadrature_abscissa )[0];

        for ( int j = 0; j < test_evals->size(); ++j )
        {
            linear_form_test( 0, 3 * j ) = ( *test_evals )[j].second[0];
            linear_form_test( 1, 3 * j + 1 ) = ( *test_evals )[j].second[0];
            linear_form_test( 2, 3 * j + 2 ) = ( *test_evals )[j].second[0];
        }
        if ( linear_form_test_indices.size() == 0 )
        {
            auto indices = surface_domain->ActiveIndex( trial_quadrature_abscissa );
            for ( auto i : indices )
            {
                linear_form_test_indices.push_back( 3 * i );
                linear_form_test_indices.push_back( 3 * i + 1 );
                linear_form_test_indices.push_back( 3 * i + 2 );
            }
        }
    }

protected:
    std::vector<Eigen::Triplet<T>> _rhs;
    const LoadFunctor _tractionFunctor;
    int _size;
};

template <int N, typename T, NeumannBoundaryType nt>
void NeumannBoundaryVisitor<N, T, nt>::LocalAssemble( Element<1, N, T>* g, const QuadratureRule<T>& quadrature_rule, const KnotSpan& knot_span )
{
    auto edge = dynamic_cast<Edge<N, T>*>( g );
    auto domain = g->GetDomain();
    auto surface_domain = edge->Parent( 0 ).lock()->GetDomain();
    _size = N * surface_domain->GetDof();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature( knot_span, quadrature_points );

    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> linear_form_test_indices;
    std::vector<Matrix> linear_form_test( num_of_quadrature ), linear_form_value( num_of_quadrature );
    std::vector<T> weights;
    std::cout << "start assemble neumman boundary.\n";
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
template <int N, typename T, NeumannBoundaryType nt>
void NeumannBoundaryVisitor<N, T, nt>::NeumannBoundaryAssembler( Eigen::SparseMatrix<T>& sparse_matrix ) const
{
    auto row_set = Accessory::RowIndicesSet( _rhs );
    this->VectorAssembler( _size, _rhs, sparse_matrix );
}
