//
// Created by miaodi on 25/12/2016.
//

#include "BsplineBasis.h"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/multiprecision/gmp.hpp>

template <typename T>
BsplineBasis<T>::BsplineBasis()
{
}

template <typename T>
BsplineBasis<T>::BsplineBasis( KnotVector<T> target ) : _basisKnot( target )
{
}

template <typename T>
int BsplineBasis<T>::GetDegree() const
{
    return _basisKnot.GetDegree();
}

template <typename T>
int BsplineBasis<T>::GetDof() const
{
    return _basisKnot.GetSize() - _basisKnot.GetDegree() - 1;
}

template <typename T>
int BsplineBasis<T>::FindSpan( const T& u ) const
{
    return _basisKnot.FindSpan( u );
}

template <typename T>
bool BsplineBasis<T>::IsActive( const int i, const T u ) const
{
    vector supp = Support( i );
    return ( u >= supp( 0 ) ) && ( u < supp( 1 ) ) ? true : false;
}

template <typename T>
T BsplineBasis<T>::EvalSingle( const T& u, const int n, const int i ) const
{
    int p = GetDegree();
    T* ders;
    T** N;
    T* ND;
    N = new T*[p + 1];
    for ( int k = 0; k < p + 1; k++ )
        N[k] = new T[p + 1];
    ND = new T[i + 1];
    ders = new T[i + 1];
    if ( u < _basisKnot[n] || u >= _basisKnot[n + p + 1] )
    {
        for ( int k = 0; k <= i; k++ )
            ders[k] = 0;
        T der = ders[i];
        delete[] ders;
        for ( int k = 0; k < p + 1; k++ )
            delete N[k];
        delete[] N;
        delete[] ND;
        return der;
    }
    for ( int j = 0; j <= p; j++ )
    {
        if ( u >= _basisKnot[n + j] && u < _basisKnot[n + j + 1] )
            N[j][0] = 1;
        else
            N[j][0] = 0;
    }
    T saved;
    for ( int k = 1; k <= p; k++ )
    {
        if ( N[0][k - 1] == 0.0 )
            saved = 0;
        else
            saved = ( ( u - _basisKnot[n] ) * N[0][k - 1] ) / ( _basisKnot[n + k] - _basisKnot[n] );
        for ( int j = 0; j < p - k + 1; j++ )
        {
            T _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + k + 1];
            if ( N[j + 1][k - 1] == 0 )
            {
                N[j][k] = saved;
                saved = 0;
            }
            else
            {
                T temp = 0;
                if ( _basisKnotright != _basisKnotleft )
                    temp = N[j + 1][k - 1] / ( _basisKnotright - _basisKnotleft );
                N[j][k] = saved + ( _basisKnotright - u ) * temp;
                saved = ( u - _basisKnotleft ) * temp;
            }
        }
    }
    ders[0] = N[0][p];
    for ( int k = 1; k <= i; k++ )
    {
        for ( int j = 0; j <= k; j++ )
            ND[j] = N[j][p - k];
        for ( int jj = 1; jj <= k; jj++ )
        {
            if ( ND[0] == 0.0 )
                saved = 0;
            else
                saved = ND[0] / ( _basisKnot[n + p - k + jj] - _basisKnot[n] );
            for ( int j = 0; j < k - jj + 1; j++ )
            {
                T _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + p + 1];
                if ( ND[j + 1] == 0 )
                {
                    ND[j] = ( p - k + jj ) * saved;
                    saved = 0;
                }
                else
                {
                    T temp = 0;
                    if ( _basisKnotright != _basisKnotleft )
                        temp = ND[j + 1] / ( _basisKnotright - _basisKnotleft );
                    ND[j] = ( p - k + jj ) * ( saved - temp );
                    saved = temp;
                }
            }
        }
        ders[k] = ND[0];
    }
    T der = ders[i];
    delete[] ders;
    for ( int k = 0; k < p + 1; k++ )
        delete N[k];
    delete[] N;
    delete[] ND;
    return der;
}

template <typename T>
typename BsplineBasis<T>::BasisFunValDerAllList_ptr BsplineBasis<T>::BezierDual( const T& u ) const
{
    int degree = _basisKnot.GetDegree();
    vector span = InSpan( u );
    T uPara = ( u - span( 0 ) ) / ( span( 1 ) - span( 0 ) );
    auto bernstein = Accessory::AllBernstein( degree, uPara );
    Eigen::Map<vector> bernsteinVector( bernstein.data(), bernstein.size() );
    // if ( !_complete_dual )
    // {
    //     int spanNum = _basisKnot.SpanNum( u );
    //     int firstIndex = FirstActive( u );
    //     vector weight = _basisWeight.block( spanNum, firstIndex, 1, degree + 1 ).transpose();
    //     vector dual = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>( weight.asDiagonal() ) *
    //                   _reconstruction[spanNum].transpose() * _gramianInv * bernsteinVector / ( span( 1 ) - span( 0 ) );
    //     BasisFunValDerAll aaa{0, std::vector<T>( 1, 0 )};
    //     BasisFunValDerAllList_ptr result( new BasisFunValDerAllList( degree + 1, aaa ) );
    //     for ( int ii = 0; ii != result->size(); ii++ )
    //         ( *result )[ii].second[0] = dual( ii );
    //     for ( int ii = 0; ii != result->size(); ++ii )
    //     {
    //         ( *result )[ii].first = firstIndex + ii;
    //     }
    //     return result;
    // }
    // else
    // {
    //     int spanNum = _basisKnot.SpanNum( u );
    //     vector dual = _localWeightContainer[spanNum].second.transpose() * _reconstruction[spanNum].transpose() *
    //                   _gramianInv * bernsteinVector / ( span( 1 ) - span( 0 ) );
    //     BasisFunValDerAll aaa{0, std::vector<T>( 1, 0 )};
    //     BasisFunValDerAllList_ptr result( new BasisFunValDerAllList( _localWeightContainer[spanNum].second.cols(), aaa ) );
    //     for ( int ii = 0; ii != result->size(); ii++ )
    //         ( *result )[ii].second[0] = dual( ii );
    //     for ( int ii = 0; ii != result->size(); ++ii )
    //     {
    //         ( *result )[ii].first = _localWeightContainer[spanNum].first + ii;
    //     }
    //     return result;
    // }

    int spanNum = _basisKnot.SpanNum( u );
    vector dual = _dualBasis._localWeightContainer[spanNum].second.transpose() * _reconstruction[spanNum].transpose() *
                  _gramianInv * bernsteinVector / ( span( 1 ) - span( 0 ) );
    BasisFunValDerAll aaa{0, std::vector<T>( 1, 0 )};
    BasisFunValDerAllList_ptr result( new BasisFunValDerAllList( _dualBasis._localWeightContainer[spanNum].second.cols(), aaa ) );
    for ( int ii = 0; ii != result->size(); ii++ )
        ( *result )[ii].second[0] = dual( ii );
    for ( int ii = 0; ii != result->size(); ++ii )
    {
        ( *result )[ii].first = _dualBasis._localWeightContainer[spanNum].first + ii;
    }
    return result;
}

template <typename T>
typename BsplineBasis<T>::BasisFunValDerAllList_ptr BsplineBasis<T>::EvalCodimensionBezierDual( const T& u ) const
{
    auto evals = BezierDual( u );
    int dof = GetDof();
    for ( auto& i : *evals )
    {
        if ( i.first == 0 )
        {
            i.first = 0;
            // i.second[0] = 0;
        }
        else if ( i.first == dof - 1 )
        {
            i.first = dof - 3;
            // i.second[0] = 0;
        }
        else
        {
            i.first = i.first - 1;
        }
    }
    return evals;
}

template <typename T>
void BsplineBasis<T>::BezierDualInitialize()
{
    int degree = _basisKnot.GetDegree();
    _reconstruction = *Accessory::BezierReconstruction<T>( _basisKnot );
    _gramianInv = Accessory::GramianInverse<T>( degree );

    // if ( !_complete_dual )
    // {
    //     _basisWeight = std::move( *BasisWeight() );
    // }
    // else
    // {
    //     auto assemble_vecs = BasisAssemblyVecs();
    //     int polynomial_completeness = degree - 1;
    //     auto lhs_rhs = LhsRhsAssembler( polynomial_completeness );
    //     auto sp_assemble = Accessory::SpVecToSpMat( assemble_vecs.begin(), assemble_vecs.end() );
    //     matrix kernel_matrix = lhs_rhs.second - sp_assemble * lhs_rhs.first;
    //     auto spans = _basisKnot.KnotSpans();
    //     std::vector<Eigen::SparseVector<T, Eigen::ColMajor>> kernel_basis_container;
    //     std::vector<Eigen::Triplet<T>> kernel_weight_triplets;
    //     for ( int i = 0; i < assemble_vecs.size(); ++i )
    //     {
    //         auto kernel_basis = Accessory::OrthonormalSpVec( assemble_vecs[i] );
    //         if ( kernel_basis.size() > 0 )
    //         {
    //             int non_zeros = assemble_vecs[i].nonZeros();
    //             const auto inner_IndexPtr = assemble_vecs[i].innerIndexPtr();
    //             int anchor_element = *( inner_IndexPtr + non_zeros / 2 ) / ( degree + 1 );
    //             auto eval = EvalDerAll( ( spans[anchor_element].first + spans[anchor_element].second ) / 2, 0 );
    //             std::vector<int> indices;
    //             for ( const auto& j : *eval )
    //             {
    //                 indices.push_back( j.first );
    //             }

    //             auto selected_dofs = Accessory::NClosestDof( indices, i, polynomial_completeness + 1 );
    //             Eigen::Ref<matrix> selected_basis =
    //                 lhs_rhs.first.block( selected_dofs[0], 0, polynomial_completeness + 1, polynomial_completeness + 1 );
    //             auto sp_kernel_basis = Accessory::SpVecToSpMat( kernel_basis.begin(), kernel_basis.end() );
    //             matrix local_rhs = kernel_matrix.transpose() * sp_kernel_basis;
    //             matrix local_kernel_weight = selected_basis.transpose().fullPivLu().solve( local_rhs );

    //             for ( int it_x = 0; it_x < local_kernel_weight.rows(); ++it_x )
    //             {
    //                 for ( int it_y = 0; it_y < local_kernel_weight.cols(); ++it_y )
    //                 {
    //                     kernel_weight_triplets.emplace_back( Eigen::Triplet<T>(
    //                         kernel_basis_container.size() + it_y, selected_dofs[0] + it_x, local_kernel_weight( it_x, it_y ) ) );
    //                 }
    //             }
    //             std::copy( kernel_basis.begin(), kernel_basis.end(), std::back_inserter( kernel_basis_container ) );
    //         }
    //     }
    //     Eigen::SparseMatrix<T> kernel_weight;
    //     const int dof = _basisKnot.GetDOF();
    //     kernel_weight.resize( kernel_basis_container.size(), dof );
    //     kernel_weight.setFromTriplets( kernel_weight_triplets.begin(), kernel_weight_triplets.end() );
    //     auto sp_kernel = Accessory::SpVecToSpMat( kernel_basis_container.begin(), kernel_basis_container.end() );
    //     _basisWeight = std::move( matrix( sp_kernel * kernel_weight + sp_assemble ) );
    //     std::cout << _basisWeight << std::endl;
    //     for ( int i = 0; i < spans.size(); ++i )
    //     {
    //         int start_dof, end_dof, start_bezier_dof;
    //         start_bezier_dof = i * ( degree + 1 );
    //         T mid_u = ( spans[i].first + spans[i].second ) / 2;
    //         start_dof = FirstActive( mid_u );
    //         end_dof = start_dof;
    //         while ( start_dof - 1 >= 0 && _basisWeight.block( start_bezier_dof, start_dof - 1, degree + 1, 1 ).norm() > 0 )
    //         {
    //             start_dof--;
    //         }
    //         while ( end_dof < dof && _basisWeight.block( start_bezier_dof, end_dof, degree + 1, 1 ).norm() > 0 )
    //         {
    //             end_dof++;
    //         }
    //         _localWeightContainer.emplace_back(
    //             std::make_pair( start_dof, static_cast<Eigen::Ref<matrix>>( _basisWeight.block(
    //                                            start_bezier_dof, start_dof, degree + 1, end_dof - start_dof ) ) ) );
    //     }
    // }

    _dualBasis._basisKnot = &( this->_basisKnot );
    _dualBasis._codimension = 0;
    _dualBasis.Initialization();
}

// Reduce the order of first two and last two elements by one (Serve as the Lagrange multiplier). The weights for boundary basis are computed.
template <typename T>
void BsplineBasis<T>::ModifyBoundaryInitialize()
{
    // p >= 1 and num of elements >= 5
    const int degree = _basisKnot.GetDegree();
    const int dof = _basisKnot.GetDOF();
    auto spans = _basisKnot.KnotSpans();
    const int elements = spans.size();
    ASSERT( degree >= 1, "The polynomial degree is too low for the modification." );
    ASSERT( elements >= 5, "There must be at least 5 elements for the modification." );

    const int boundary_degree = degree - 1;
    KnotVector<T> first_element_knot, second_element_knot, second_last_element_knot, last_element_knot;
    first_element_knot.InitClosed( boundary_degree, spans[0].first, spans[0].second );
    second_element_knot.InitClosed( boundary_degree, spans[1].first, spans[1].second );
    second_last_element_knot.InitClosed( boundary_degree, spans[elements - 2].first, spans[elements - 2].second );
    last_element_knot.InitClosed( boundary_degree, spans[elements - 1].first, spans[elements - 1].second );
    BsplineBasis<T> first_element( first_element_knot ), second_element( second_element_knot ),
        second_last_element( second_last_element_knot ), last_element( last_element_knot );

    auto second_end_eval = second_element.EvalDerAll( spans[1].second, boundary_degree );
    auto second_begin_eval = second_element.EvalDerAll( spans[1].first, boundary_degree );
    auto first_end_eval = first_element.EvalDerAll( spans[0].second, boundary_degree );

    auto second_last_begin_eval = second_last_element.EvalDerAll( spans[elements - 2].first, boundary_degree );
    auto second_last_end_eval = second_last_element.EvalDerAll( spans[elements - 2].second, boundary_degree );
    auto last_begin_eval = last_element.EvalDerAll( spans[elements - 1].first, boundary_degree );
    matrix second_end( boundary_degree + 1, boundary_degree + 1 ),
        second_begin( boundary_degree + 1, boundary_degree + 1 ), first_end( boundary_degree + 1, boundary_degree + 1 );
    matrix second_last_begin( boundary_degree + 1, boundary_degree + 1 ),
        second_last_end( boundary_degree + 1, boundary_degree + 1 ), last_begin( boundary_degree + 1, boundary_degree + 1 );
    for ( int i = 0; i < boundary_degree + 1; i++ )
    {
        for ( int j = 0; j < boundary_degree + 1; j++ )
        {
            second_end( i, j ) = ( *second_end_eval )[j].second[i];
            second_begin( i, j ) = ( *second_begin_eval )[j].second[i];
            first_end( i, j ) = ( *first_end_eval )[j].second[i];

            second_last_begin( i, j ) = ( *second_last_begin_eval )[j].second[i];
            second_last_end( i, j ) = ( *second_last_end_eval )[j].second[i];
            last_begin( i, j ) = ( *last_begin_eval )[j].second[i];
        }
    }
    std::vector<vector> front_basis_eval( boundary_degree + 1 );
    std::vector<vector> back_basis_eval( boundary_degree + 1 );
    for ( int i = 0; i < boundary_degree + 1; i++ )
    {
        front_basis_eval[i].resize( boundary_degree + 1 );
        back_basis_eval[i].resize( boundary_degree + 1 );
        for ( int j = 0; j < boundary_degree + 1; j++ )
        {
            front_basis_eval[i]( j ) = this->EvalSingle( spans[1].second, 2 + i, j );
            back_basis_eval[i]( j ) = this->EvalSingle( spans[elements - 2].first, dof - 3 - boundary_degree + i, j );
        }
    }
    std::vector<vector> weights_in_second_element( boundary_degree + 1 );
    std::vector<vector> weights_in_second_last_element( boundary_degree + 1 );
    for ( int i = 0; i < boundary_degree + 1; i++ )
    {
        weights_in_second_element[i] = second_end.fullPivLu().solve( front_basis_eval[i] );
        weights_in_second_last_element[i] = second_last_begin.fullPivLu().solve( back_basis_eval[i] );
    }
    std::vector<vector> weights_in_first_element( boundary_degree + 1 );
    std::vector<vector> weights_in_last_element( boundary_degree + 1 );
    for ( int i = 0; i < boundary_degree + 1; i++ )
    {
        weights_in_first_element[i] = first_end.fullPivLu().solve( second_begin * weights_in_second_element[i] );
        weights_in_last_element[i] = last_begin.fullPivLu().solve( second_last_end * weights_in_second_last_element[i] );
    }
    _basisWeight.resize( boundary_degree + 1, 4 * ( boundary_degree + 1 ) );
    for ( int i = 0; i < boundary_degree + 1; i++ )
    {
        _basisWeight.col( i ) = weights_in_first_element[i];
        _basisWeight.col( i + boundary_degree + 1 ) = weights_in_second_element[i];
        _basisWeight.col( i + 2 * ( boundary_degree + 1 ) ) = weights_in_second_last_element[i];
        _basisWeight.col( i + 3 * ( boundary_degree + 1 ) ) = weights_in_last_element[i];
    }
}

template <typename T>
std::unique_ptr<typename BsplineBasis<T>::matrix> BsplineBasis<T>::BasisWeight() const
{
    using QuadList = typename QuadratureRule<T>::QuadList;
    std::unique_ptr<matrix> result( new matrix );
    int dof = _basisKnot.GetDOF();
    auto spans = _basisKnot.KnotEigenSpans();
    int elements = spans.size();
    int degree = _basisKnot.GetDegree();
    result->resize( elements, dof );
    result->setZero();
    QuadratureRule<T> quadrature( ( degree + 1 ) / 2 + ( degree + 1 ) % 2 );
    int num = 0;
    for ( auto& i : spans )
    {
        QuadList quadList;
        quadrature.MapToQuadrature( i, quadList );
        for ( auto& j : quadList )
        {
            auto evals = EvalDerAll( j.first( 0 ), 0 );
            for ( auto& k : *evals )
            {
                ( *result )( num, k.first ) += j.second * k.second[0];
            }
        }
        num++;
    }
    vector sumWeight( dof );
    sumWeight.setZero();
    for ( int i = 0; i < dof; i++ )
    {
        sumWeight( i ) = result->col( i ).sum();
    }
    for ( int i = 0; i < dof; i++ )
    {
        result->col( i ) /= sumWeight( i );
    }
    return result;
}

// Assembly matrix A for polynomial completeness dual.
template <typename T>
std::vector<Eigen::SparseVector<T>> BsplineBasis<T>::BasisAssemblyVecs() const
{
    // basic variables
    int dof = _basisKnot.GetDOF();
    auto spans = _basisKnot.KnotSpans();
    int elements = spans.size();
    int degree = _basisKnot.GetDegree();
    int dof_in_element = degree + 1;

    std::vector<Eigen::SparseVector<T>> result;
    for ( int i = 0; i < dof; ++i )
    {
        result.push_back( Eigen::SparseVector<T>() );
        result[i].resize( elements * dof_in_element );
    }
    for ( const auto& i : spans )
    {
        T u = ( i.first + i.second ) / 2;
        int spanNum = _basisKnot.SpanNum( u );
        int firstIndex = FirstActive( u );
        for ( int j = 0; j < dof_in_element; ++j )
        {
            result[firstIndex + j].coeffRef( dof_in_element * spanNum + j ) = 1;
        }
    }
    for ( auto& i : result )
    {
        i /= i.nonZeros();
    }
    return result;
}

// lhs and rhs assembler for polynomial completeness dual.
template <typename T>
std::pair<typename BsplineBasis<T>::matrix, typename BsplineBasis<T>::matrix> BsplineBasis<T>::LhsRhsAssembler( int degree_of_completeness ) const
{
    using namespace boost::math;
    using QuadList = typename QuadratureRule<T>::QuadList;

    int degree = _basisKnot.GetDegree();
    int dof = _basisKnot.GetDOF();
    ASSERT( degree_of_completeness <= degree, "polynomial completeness is higher than polynomial degree." );
    auto spans = _basisKnot.KnotEigenSpans();

    QuadratureRule<T> quadrature( degree + 1 );
    matrix rhs( spans.size() * ( degree + 1 ), degree_of_completeness + 1 );
    matrix lhs( dof, degree_of_completeness + 1 );
    rhs.setZero();
    lhs.setZero();
    const T u_b = *_basisKnot.cbegin();
    const T u_e = *( _basisKnot.cend() - 1 );
    auto f = [u_e, u_b]( T u ) { return 2 * ( u - u_b ) / ( u_e - u_b ) - 1; };
    for ( int i = 0; i < spans.size(); ++i )
    {
        QuadList quadList;
        quadrature.MapToQuadrature( spans[i], quadList );
        for ( auto& j : quadList )
        {
            T u = j.first( 0 );
            auto evals = EvalDerAll( u, 0 );
            matrix temp( 1, ( degree + 1 ) );
            for ( int j = 0; j < evals->size(); ++j )
            {
                temp( 0, j ) = ( *evals )[j].second[0];
            }
            matrix polynomials( 1, degree_of_completeness + 1 );
            for ( int i = 0; i <= degree_of_completeness; i++ )
            {
                // polynomials( 0, i ) = legendre_p( i, f( u ) );
                polynomials( 0, i ) = pow( u, i );
            }
            rhs.block( ( degree + 1 ) * i, 0, ( degree + 1 ), degree_of_completeness + 1 ) += j.second * temp.transpose() * polynomials;
            lhs.block( ( *evals )[0].first, 0, ( degree + 1 ), degree_of_completeness + 1 ) +=
                j.second * temp.transpose() * polynomials;
        }
    }
    return std::make_pair( std::move( lhs ), std::move( rhs ) );
}

template <typename T>
typename BsplineBasis<T>::BasisFunValDerAllList_ptr BsplineBasis<T>::EvalModifiedDerAll( const T& u, int i ) const
{
    const int degree = _basisKnot.GetDegree();
    const int dof = _basisKnot.GetDOF();
    const int element_num = this->FindSpan( u ) - degree;
    const int boundary_degree = degree - 1;
    auto spans = _basisKnot.KnotSpans();
    const int elements = spans.size();

    if ( element_num == 0 )
    {
        KnotVector<T> first_element_knot;
        first_element_knot.InitClosed( boundary_degree, spans[0].first, spans[0].second );
        BsplineBasis<T> first_element( first_element_knot );
        auto res = first_element.EvalDerAll( u, i );
        matrix matrix_form( i + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                matrix_form( k, j ) = ( *res )[j].second[k];
            }
        }
        matrix res_matrix_form = matrix_form * _basisWeight.block( 0, 0, boundary_degree + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                ( *res )[j].second[k] = res_matrix_form( k, j );
            }
        }
        return res;
    }
    else if ( element_num == 1 )
    {
        KnotVector<T> second_element_knot;
        second_element_knot.InitClosed( boundary_degree, spans[1].first, spans[1].second );
        BsplineBasis<T> second_element( second_element_knot );
        auto res = second_element.EvalDerAll( u, i );
        matrix matrix_form( i + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                matrix_form( k, j ) = ( *res )[j].second[k];
            }
        }
        matrix res_matrix_form =
            matrix_form * _basisWeight.block( 0, boundary_degree + 1, boundary_degree + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                ( *res )[j].second[k] = res_matrix_form( k, j );
            }
        }
        return res;
    }
    else if ( element_num == elements - 2 )
    {
        KnotVector<T> second_last_element_knot;
        second_last_element_knot.InitClosed( boundary_degree, spans[elements - 2].first, spans[elements - 2].second );
        BsplineBasis<T> second_last_element( second_last_element_knot );
        auto res = second_last_element.EvalDerAll( u, i );
        matrix matrix_form( boundary_degree + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                matrix_form( k, j ) = ( *res )[j].second[k];
            }
        }
        matrix res_matrix_form =
            matrix_form * _basisWeight.block( 0, 2 * ( boundary_degree + 1 ), boundary_degree + 1, boundary_degree + 1 );
        for ( int j = 0; j < boundary_degree + 1; j++ )
        {
            ( *res )[j].first = dof - 5 - boundary_degree + j;
            for ( int k = 0; k <= i; k++ )
            {
                ( *res )[j].second[k] = res_matrix_form( k, j );
            }
        }
        return res;
    }
    else if ( element_num == elements - 1 )
    {
        KnotVector<T> last_element_knot;
        last_element_knot.InitClosed( boundary_degree, spans[elements - 1].first, spans[elements - 1].second );
        BsplineBasis<T> last_element( last_element_knot );
        auto res = last_element.EvalDerAll( u, i );
        matrix matrix_form( boundary_degree + 1, boundary_degree + 1 );
        for ( int k = 0; k <= i; k++ )
        {
            for ( int j = 0; j < boundary_degree + 1; j++ )
            {
                matrix_form( k, j ) = ( *res )[j].second[k];
            }
        }
        matrix res_matrix_form =
            matrix_form * _basisWeight.block( 0, 3 * ( boundary_degree + 1 ), boundary_degree + 1, boundary_degree + 1 );
        for ( int j = 0; j < boundary_degree + 1; j++ )
        {
            ( *res )[j].first = dof - 5 - boundary_degree + j;
            for ( int k = 0; k <= i; k++ )
            {
                ( *res )[j].second[k] = res_matrix_form( k, j );
            }
        }
        return res;
    }
    else
    {
        auto res = this->EvalDerAll( u, i );
        for ( auto& i : *res )
        {
            i.first -= 2;
        }
        return res;
    }
}

template <typename T>
typename BsplineBasis<T>::BasisFunValPac_ptr BsplineBasis<T>::Eval( const T& u, const int i ) const
{
    const int dof = GetDof();
    const int deg = GetDegree();
    matrix ders;
    T* left = new T[2 * ( deg + 1 )];
    T* right = &left[deg + 1];
    matrix ndu( deg + 1, deg + 1 );
    T saved, temp;
    int j, r;
    int span = FindSpan( u );
    ders.resize( i + 1, deg + 1 );

    ndu( 0, 0 ) = T( 1 );
    for ( j = 1; j <= deg; j++ )
    {
        left[j] = u - _basisKnot[span + 1 - j];
        right[j] = _basisKnot[span + j] - u;
        saved = T( 0 );

        for ( r = 0; r < j; r++ )
        {
            // Lower triangle
            ndu( j, r ) = right[r + 1] + left[j - r];
            temp = ndu( r, j - 1 ) / ndu( j, r );
            // _basisKnotpper triangle
            ndu( r, j ) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        ndu( j, j ) = saved;
    }

    for ( j = deg; j >= 0; --j )
        ders( 0, j ) = ndu( j, deg );

    // Compute the derivatives
    matrix a( deg + 1, deg + 1 );
    for ( r = 0; r <= deg; r++ )
    {
        int s1, s2;
        s1 = 0;
        s2 = 1; // alternate rows in array a
        a( 0, 0 ) = T( 1 );
        // Compute the kth derivative
        for ( int k = 1; k <= i; k++ )
        {
            T d{0};
            int rk{r - k}, pk{deg - k}, j1{0}, j2{0};

            if ( r >= k )
            {
                a( s2, 0 ) = a( s1, 0 ) / ndu( pk + 1, rk );
                d = a( s2, 0 ) * ndu( rk, pk );
            }

            if ( rk >= -1 )
            {
                j1 = 1;
            }
            else
            {
                j1 = -rk;
            }

            if ( r - 1 <= pk )
            {
                j2 = k - 1;
            }
            else
            {
                j2 = deg - r;
            }

            for ( j = j1; j <= j2; j++ )
            {
                a( s2, j ) = ( a( s1, j ) - a( s1, j - 1 ) ) / ndu( pk + 1, rk + j );
                d += a( s2, j ) * ndu( rk + j, pk );
            }

            if ( r <= pk )
            {
                a( s2, k ) = -a( s1, k - 1 ) / ndu( pk + 1, r );
                d += a( s2, k ) * ndu( r, pk );
            }
            ders( k, r ) = d;
            j = s1;
            s1 = s2;
            s2 = j; // Switch rows
        }
    }

    // Multiply through by the correct factors
    r = deg;
    for ( int k = 1; k <= i; k++ )
    {
        for ( j = deg; j >= 0; --j )
            ders( k, j ) *= T( r );
        r *= deg - k;
    }
    delete[] left;
    BasisFunValPac_ptr result( new BasisFunValPac );
    int firstIndex = FirstActive( u );
    for ( int ii = 0; ii != ders.cols(); ++ii )
    {
        result->push_back( BasisFunVal( firstIndex + ii, ders( i, ii ) ) );
    }
    return result;
}

template <typename T>
typename BsplineBasis<T>::vector BsplineBasis<T>::InSpan( const T& u ) const
{
    auto span = FindSpan( u );
    vector res( 2 );
    res << _basisKnot[span], _basisKnot[span + 1];
    return res;
}

template <typename T>
typename BsplineBasis<T>::BasisFunValDerAllList_ptr BsplineBasis<T>::EvalDerAll( const T& u, int i ) const
{
    const int deg = GetDegree();
    BasisFunValDerAll aaa{0, std::vector<T>( i + 1, 0 )};
    BasisFunValDerAllList_ptr ders( new BasisFunValDerAllList( deg + 1, aaa ) );
    T* left = new T[2 * ( deg + 1 )];
    T* right = &left[deg + 1];
    matrix ndu( deg + 1, deg + 1 );
    T saved, temp;
    int j, r;
    int span = FindSpan( u );
    ndu( 0, 0 ) = T( 1 );
    for ( j = 1; j <= deg; j++ )
    {
        left[j] = u - _basisKnot[span + 1 - j];
        right[j] = _basisKnot[span + j] - u;
        saved = T( 0 );

        for ( r = 0; r < j; r++ )
        {
            // Lower triangle
            ndu( j, r ) = right[r + 1] + left[j - r];
            temp = ndu( r, j - 1 ) / ndu( j, r );
            // _basisKnotpper triangle
            ndu( r, j ) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        ndu( j, j ) = saved;
    }
    for ( j = deg; j >= 0; --j )
        ( *ders )[j].second[0] = ndu( j, deg );

    // Compute the derivatives
    matrix a( deg + 1, deg + 1 );
    for ( r = 0; r <= deg; r++ )
    {
        int s1, s2;
        s1 = 0;
        s2 = 1; // alternate rows in array a
        a( 0, 0 ) = T( 1 );
        // Compute the kth derivative
        for ( int k = 1; k <= i; k++ )
        {
            T d{0};
            int rk{r - k}, pk{deg - k}, j1, j2;

            if ( r >= k )
            {
                a( s2, 0 ) = a( s1, 0 ) / ndu( pk + 1, rk );
                d = a( s2, 0 ) * ndu( rk, pk );
            }

            if ( rk >= -1 )
            {
                j1 = 1;
            }
            else
            {
                j1 = -rk;
            }

            if ( r - 1 <= pk )
            {
                j2 = k - 1;
            }
            else
            {
                j2 = deg - r;
            }

            for ( j = j1; j <= j2; j++ )
            {
                a( s2, j ) = ( a( s1, j ) - a( s1, j - 1 ) ) / ndu( pk + 1, rk + j );
                d += a( s2, j ) * ndu( rk + j, pk );
            }

            if ( r <= pk )
            {
                a( s2, k ) = -a( s1, k - 1 ) / ndu( pk + 1, r );
                d += a( s2, k ) * ndu( r, pk );
            }
            ( *ders )[r].second[k] = d;
            j = s1;
            s1 = s2;
            s2 = j; // Switch rows
        }
    }

    // Multiply through by the correct factors
    r = deg;
    for ( int k = 1; k <= i; k++ )
    {
        for ( j = deg; j >= 0; --j )
            ( *ders )[j].second[k] *= T( r );
        r *= deg - k;
    }
    delete[] left;

    int firstIndex = FirstActive( u );
    for ( int ii = 0; ii != ders->size(); ++ii )
    {
        ( *ders )[ii].first = firstIndex + ii;
    }
    return ders;
}

template <typename T>
typename BsplineBasis<T>::vector BsplineBasis<T>::Support( const int i ) const
{
    const int deg = GetDegree();
    ASSERT( i < GetDof(), "Invalid index of basis function." );
    vector res( 2 );
    res << _basisKnot[i], _basisKnot[i + deg + 1];
    return res;
}

// template class BsplineBasis<long double>;
template class BsplineBasis<double>;
// template class BsplineBasis<float>;