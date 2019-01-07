//
// Created by miaodi on 26/12/2016.
//

#include "TensorBsplineBasis.h"
#include <boost/multiprecision/gmp.hpp>

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis()
{
    for ( int i = 0; i < d; ++i )
        _basis[i] = BsplineBasis<T>();
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const BsplineBasis<T>& baseX )
{
    ASSERT( d == 1, "Invalid dimension." );
    _basis[0] = baseX;
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY )
{
    ASSERT( d == 2, "Invalid dimension." );
    _basis[0] = baseX;
    _basis[1] = baseY;
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY, const BsplineBasis<T>& baseZ )
{
    ASSERT( d == 3, "Invalid dimension." );
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const KnotVector<T>& kVX )
{
    ASSERT( d == 1, "Invalid dimension." );
    _basis[0] = BsplineBasis<T>( kVX );
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const KnotVector<T>& kVX, const KnotVector<T>& kVY )
{
    ASSERT( d == 2, "Invalid dimension." );
    _basis[0] = BsplineBasis<T>( kVX );
    _basis[1] = BsplineBasis<T>( kVY );
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const KnotVector<T>& kVX, const KnotVector<T>& kVY, const KnotVector<T>& kVZ )
{
    ASSERT( d == 3, "Invalid dimension." );
    _basis[0] = BsplineBasis<T>( kVX );
    _basis[1] = BsplineBasis<T>( kVY );
    _basis[2] = BsplineBasis<T>( kVZ );
}

template <int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis( const std::vector<KnotVector<T>>& knotVectors )
{
    ASSERT( d == knotVectors.size(), "Invalid number of knot-vectors given." );
    for ( int i = 0; i != d; ++i )
        _basis[i] = BsplineBasis<T>( knotVectors[i] );
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::GetDegree( const int i ) const
{
    return _basis[i].GetDegree();
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::GetDof() const
{
    int dof = 1;
    for ( int i = 0; i != d; ++i )
        dof *= _basis[i].GetDof();
    return dof;
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::GetDof( const int i ) const
{
    return _basis[i].GetDof();
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::NumActive( const int& i ) const
{
    return _basis[i].NumActive();
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::NumActive() const
{
    int active = 1;
    for ( int i = 0; i != d; ++i )
        active *= _basis[i].NumActive();
    return active;
}

template <int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr TensorBsplineBasis<d, T>::EvalTensor( const TensorBsplineBasis::vector& u,
                                                                                            const TensorBsplineBasis::DiffPattern& i ) const
{
    ASSERT( ( u.size() == d ) && ( i.size() == d ), "Invalid input vector size." );
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    std::array<BasisFunValPac_ptr, d> OneDResult;
    for ( int direction = 0; direction != d; ++direction )
    {
        OneDResult[direction] = _basis[direction].Eval( u( direction ), i[direction] );
        endPerIndex.push_back( OneDResult[direction]->size() );
    }
    std::vector<int> MultiIndex( d );
    std::vector<T> Value( d );
    BasisFunValPac_ptr Result( new BasisFunValPac );

    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;

    recursive = [this, &OneDResult, &MultiIndex, &Value, &Result, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            T result = 1;
            for ( int ii = 0; ii < d; ii++ )
                result *= Value[ii];
            Result->push_back( BasisFunVal( Index( MultiIndex ), result ) );
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                Value[direction] = ( *OneDResult[direction] )[indexes[direction]].second;
                MultiIndex[direction] = ( *OneDResult[direction] )[indexes[direction]].first;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    return Result;
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::ChangeKnots( const KnotVector<T>& knots, int direction )
{
    _basis[direction] = knots;
}

template <int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr TensorBsplineBasis<d, T>::EvalDualAllTensor( const TensorBsplineBasis::vector& u ) const
{
    ASSERT( ( u.size() == d ), "Invalid input vector size." );
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    std::vector<int> MultiIndex( d );
    std::vector<T> Value( d );
    BasisFunValDerAllList_ptr Result( new BasisFunValDerAllList );
    std::array<BasisFunValDerAllList_ptr, d> oneDResult;
    for ( int direction = 0; direction != d; ++direction )
    {
        oneDResult[direction] = _basis[direction].BezierDual( static_cast<T>( u( direction ) ) );
        endPerIndex.push_back( oneDResult[direction]->size() );
    }
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    recursive = [this, &oneDResult, &MultiIndex, &Value, &Result, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            std::vector<T> result( 1, 1 );
            for ( int ii = 0; ii < d; ii++ )
                result[0] *= Value[ii];
            Result->push_back( BasisFunValDerAll( Index( MultiIndex ), result ) );
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                Value[direction] = ( *oneDResult[direction] )[indexes[direction]].second[0];
                MultiIndex[direction] = ( *oneDResult[direction] )[indexes[direction]].first;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    return Result;
}

template <int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr TensorBsplineBasis<d, T>::EvalDerAllTensor(
    const TensorBsplineBasis::vector& u, const int i ) const
{
    ASSERT( ( u.size() == d ), "Invalid input vector size." );
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    Accessory::DifferentialPatternList differentialPatternList;
    for ( int order = 0; order <= i; ++order )
    {
        auto temp = Accessory::PartialDerPattern<d>( order );
        differentialPatternList.insert( differentialPatternList.end(), temp->begin(), temp->end() );
    }
    int derivativeAmount = differentialPatternList.size();
    BasisFunValDerAllList_ptr Result( new BasisFunValDerAllList );
    std::array<BasisFunValDerAllList_ptr, d> oneDResult;
    for ( int direction = 0; direction != d; ++direction )
    {
        oneDResult[direction] = _basis[direction].EvalDerAll( u( direction ), i );
        endPerIndex.push_back( oneDResult[direction]->size() );
    }
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    std::vector<int> multiIndex( d );
    std::vector<std::vector<T>> Values( derivativeAmount, std::vector<T>( d, 0 ) );
    recursive = [this, &derivativeAmount, &oneDResult, &multiIndex, &Values, &Result, &differentialPatternList,
                 &recursive]( std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            std::vector<T> result( derivativeAmount, 1 );
            for ( int iii = 0; iii != derivativeAmount; ++iii )
            {
                for ( int ii = 0; ii != d; ++ii )
                {
                    result[iii] *= Values[iii][ii];
                }
            }
            Result->push_back( BasisFunValDerAll( Index( multiIndex ), result ) );
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                for ( auto it_diffPart = differentialPatternList.begin(); it_diffPart != differentialPatternList.end(); ++it_diffPart )
                {
                    int diffPart_label = it_diffPart - differentialPatternList.begin();
                    Values[diffPart_label][direction] =
                        ( *oneDResult[direction] )[indexes[direction]].second[( *it_diffPart )[direction]];
                }
                multiIndex[direction] = ( *oneDResult[direction] )[indexes[direction]].first;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    return Result;
}

// orientation is the normal direction,
template <int d, typename T>
std::unique_ptr<std::vector<int>> TensorBsplineBasis<d, T>::HyperPlaneIndices( const int& orientation, const int& layer ) const
{
    ASSERT( orientation < d, "Invalid input vector size." );
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex( d, 0 );
    for ( int i = 0; i != d; ++i )
    {
        if ( i == orientation )
        {
            endPerIndex[i] = 1;
        }
        else
        {
            endPerIndex[i] = GetDof( i );
        }
    }
    std::unique_ptr<std::vector<int>> result( new std::vector<int> );
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    std::vector<int> temp( d, 0 );
    recursive = [this, &orientation, &layer, &result, &temp, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == d )
        {
            result->push_back( Index( temp ) );
        }
        else
        {
            if ( direction == orientation )
            {
                temp[direction] = layer;
                recursive( indexes, endPerIndex, direction + 1 );
            }
            else
            {
                for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
                {
                    temp[direction] = indexes[direction];
                    recursive( indexes, endPerIndex, direction + 1 );
                }
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    return result;
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::PrintEvalDerAllTensor( const TensorBsplineBasis::vector& u, const int diff ) const
{
    auto eval = EvalDerAllTensor( u, diff );
    for ( const auto& i : *eval )
    {
        std::cout << i.first << "th basis: ";
        for ( const auto& j : i.second )
        {
            std::cout << std::setprecision( std::numeric_limits<T>::digits ) << j << " ";
        }
        std::cout << std::endl;
    }
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::PrintEvalTensor( const TensorBsplineBasis::vector& u, const TensorBsplineBasis::DiffPattern& diff ) const
{
    auto eval = EvalTensor( u, diff );
    for ( const auto& i : *eval )
    {
        std::cout << i.first << "th basis: " << std::setprecision( std::numeric_limits<T>::digits ) << i.second << std::endl;
    }
}

template <int d, typename T>
std::unique_ptr<std::vector<int>> TensorBsplineBasis<d, T>::Indices() const
{
    auto dof = this->GetDof();
    auto res = std::make_unique<std::vector<int>>();
    for ( int i = 0; i < dof; ++i )
    {
        res->push_back( i );
    }
    return res;
}

// Create knot pairs that represent the south west corner and north east corner.
template <int d, typename T>
void TensorBsplineBasis<d, T>::KnotSpanGetter( TensorBsplineBasis<d, T>::KnotSpanList& knot_spans ) const
{
    knot_spans.clear();
    std::array<std::vector<std::pair<T, T>>, d> knot_span_in;
    for ( int i = 0; i < d; ++i )
    {
        knot_span_in[i] = KnotVectorGetter( i ).KnotSpans();
    }
    std::vector<int> indexes( d, 0 );
    std::vector<int> endIndex( d );
    for ( int i = 0; i != d; ++i )
    {
        endIndex[i] = knot_span_in[i].size();
    }
    std::function<void( std::vector<int>&, int )> recursive;
    vector start_knot( d ), end_knot( d );
    recursive = [this, &endIndex, &knot_spans, &knot_span_in, &start_knot, &end_knot, &recursive](
                    std::vector<int>& indexes, int direction ) {
        if ( direction == d )
        {
            knot_spans.push_back( std::make_pair( start_knot, end_knot ) );
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endIndex[direction]; indexes[direction]++ )
            {
                start_knot( direction ) = knot_span_in[direction][indexes[direction]].first;
                end_knot( direction ) = knot_span_in[direction][indexes[direction]].second;
                recursive( indexes, direction + 1 );
            }
        }
    };
    recursive( indexes, 0 );
}

template <int d, typename T>
bool TensorBsplineBasis<d, T>::InDomain( const TensorBsplineBasis<d, T>::vector& u ) const
{
    ASSERT( ( u.size() == d ), "Invalid input vector size." );
    for ( int i = 0; i < d; ++i )
    {
        if ( u( i ) < DomainStart( i ) || u( i ) > DomainEnd( i ) )
        {
            return false;
        }
    }
    return true;
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::FixOnBoundary( TensorBsplineBasis<d, T>::vector& u, T tol ) const
{
    ASSERT( ( u.size() == d ), "Invalid input vector size." );
    for ( int i = 0; i < d; ++i )
    {
        T domain_length = DomainEnd( i ) - DomainStart( i );
        if ( abs( DomainStart( i ) - u( i ) ) / domain_length < tol )
        {
            u( i ) = DomainStart( i );
        }
        if ( abs( DomainEnd( i ) - u( i ) ) / domain_length < tol )
        {
            u( i ) = DomainEnd( i );
        }
    }
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::BezierDualInitialize()
{
    for ( int direction = 0; direction < d; ++direction )
    {
        _basis[direction].BezierDualInitialize();
    }
}

template <int d, typename T>
void TensorBsplineBasis<d, T>::ModifyBoundaryInitialize()
{
    for ( int direction = 0; direction < d; ++direction )
    {
        _basis[direction].ModifyBoundaryInitialize();
    }
}

//! Return the index in each directions.
template <int d, typename T>
std::vector<int> TensorBsplineBasis<d, T>::TensorIndex( const int& m ) const
{
    ASSERT( m < GetDof(), "Input index is invalid." );
    std::vector<int> ind( d );
    int mm = m;
    /// int always >=0.
    for ( int i = static_cast<int>( d - 1 ); i >= 0; --i )
    {
        ind[i] = mm % GetDof( i );
        mm -= ind[i];
        mm /= GetDof( i );
    }
    return ind;
}

template <int d, typename T>
int TensorBsplineBasis<d, T>::Index( const std::vector<int>& ts ) const
{
    ASSERT( ts.size() == d, "Input index is invalid." );
    int index = 0;
    for ( int direction = 0; direction < d; ++direction )
    {
        index += ts[direction];
        if ( direction != d - 1 )
        {
            index *= GetDof( direction + 1 );
        }
    }
    return index;
}

template <int d, typename T>
typename TensorBsplineBasis<d, T>::matrix TensorBsplineBasis<d, T>::Support( const int& i ) const
{
    matrix res( static_cast<int>( d ), 2 );
    auto ti = TensorIndex( i );
    for ( int j = 0; j != d; ++j )
        res.row( j ) = _basis[j].Support( ti[j] );
    return res;
}

template <int d, typename T>
std::vector<int> TensorBsplineBasis<d, T>::ActiveIndex( const vector& u ) const
{
    std::vector<int> temp;
    temp.reserve( NumActive() );
    ASSERT( ( u.size() == d ), "Invalid input vector size." );
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex( d );
    std::vector<int> startIndex( d );
    for ( int i = 0; i != d; ++i )
    {
        startIndex[i] = _basis[i].FirstActive( u( i ) );
        endPerIndex[i] = _basis[i].NumActive();
    }
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    std::vector<int> multiIndex( d );
    recursive = [this, &startIndex, &temp, &multiIndex, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            temp.push_back( Index( multiIndex ) );
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                multiIndex[direction] = startIndex[direction] + indexes[direction];
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    return temp;
}

template <int d, typename T>
T TensorBsplineBasis<d, T>::EvalSingle( const vector& u, const int n, const DiffPattern& i ) const
{
    ASSERT( ( u.size() == d ) && ( i.size() == d ), "Invalid input vector size." );
    auto tensorindex = TensorIndex( n );
    T result = 1;
    for ( int direction = 0; direction != d; ++direction )
    {
        result *= _basis[direction].EvalSingle( u( direction ), tensorindex[direction], i[direction] );
    }
    return result;
}

template <typename T>
TensorBsplineBasis<0, T>::TensorBsplineBasis( const T& support ) : _basis{support}
{
}

template <typename T>
TensorBsplineBasis<0, T>::TensorBsplineBasis()
{
}

// template class TensorBsplineBasis<0, long double>;
// template class TensorBsplineBasis<1, long double>;
// template class TensorBsplineBasis<2, long double>;
// template class TensorBsplineBasis<3, long double>;

template class TensorBsplineBasis<0, double>;
template class TensorBsplineBasis<1, double>;
template class TensorBsplineBasis<2, double>;
template class TensorBsplineBasis<3, double>;

// template class TensorBsplineBasis<1, float>;
// template class TensorBsplineBasis<2, float>;
// template class TensorBsplineBasis<3, float>;
