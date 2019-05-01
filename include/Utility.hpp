//
// Created by miaodi on 23/10/2017.
//

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <KnotVector.h>
// #include <boost/math/special_functions/binomial.hpp>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <vector>

// #ifndef NDEBUG
// #define ASSERT(condition, message)                                             \
//     do                                                                         \
//     {                                                                          \
//         if (!(condition))                                                      \
//         {                                                                      \
//             std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
//                       << " line " << __LINE__ << ": " << message << std::endl; \
//             std::terminate();                                                  \
//         }                                                                      \
//     } while (false)
// #else
// #define ASSERT(condition, message) \
//     do                             \
//     {                              \
//     } while (false)
// #endif

template <int d, int N, typename T>
class PhyTensorBsplineBasis;

template <typename T>
class KnotVector;

namespace Accessory
{
using namespace Eigen;

// aligned_allocator is required by Eigen for fixed-size Eigen types
template <typename T, int N>
using ContPtsList = std::vector<Eigen::Matrix<T, N, 1>, aligned_allocator<Matrix<T, N, 1>>>;

using DifferentialPattern = std::vector<int>;
using DifferentialPatternList = std::vector<DifferentialPattern>;
using DifferentialPatternList_ptr = std::unique_ptr<DifferentialPatternList>;
template <typename T>
using ExtractionOperator = Matrix<T, Dynamic, Dynamic>;
template <typename T>
using ExtractionOperatorContainer = std::vector<ExtractionOperator<T>>;

template <typename T>
void binomialCoef( Matrix<T, Dynamic, Dynamic>& Bin )
{
    int n, k;
    // Setup the first line
    Bin( 0, 0 ) = T( 1 );
    for ( k = static_cast<int>( Bin.cols() ) - 1; k > 0; --k )
        Bin( 0, k ) = T( 0 );
    // Setup the other lines
    for ( n = 0; n < static_cast<int>( Bin.rows() ) - 1; n++ )
    {
        Bin( n + 1, 0 ) = T( 1 );
        for ( k = 1; k < static_cast<int>( Bin.cols() ); k++ )
            if ( n + 1 < k )
                Bin( n, k ) = T( 0 );
            else
                Bin( n + 1, k ) = Bin( n, k ) + Bin( n, k - 1 );
    }
}

template <typename T, int N>
void degreeElevate( int t, KnotVector<T>& U, ContPtsList<T, N>& P )
{
    ASSERT( t > 0, "Invalid geometrical information input, check size bro." );
    int i, j, k;
    auto dof = U.GetDOF();
    auto cP = P;
    auto cU = U;
    int n = dof - 1;
    int p = U.GetDegree();
    int m = n + p + 1;
    int ph = p + t;
    int ph2 = ph / 2;
    Matrix<T, Dynamic, Dynamic> bezalfs( p + t + 1, p + 1 ); // coefficients for degree elevating the Bezier segment
    std::vector<Matrix<T, N, 1>, aligned_allocator<Matrix<T, N, 1>>> bpts( p + 1 ); // pth-degree Bezier control points of the current segment
    std::vector<Matrix<T, N, 1>, aligned_allocator<Matrix<T, N, 1>>> ebpts( p + t + 1 ); // (p+t)th-degree Bezier control points of the  current segment
    std::vector<Matrix<T, N, 1>, aligned_allocator<Matrix<T, N, 1>>> Nextbpts( p - 1 ); // leftmost control points of the next Bezier segment
    std::vector<T> alphas( p - 1, T( 0 ) );                                             // knot instertion alphas.
    // Compute the binomial coefficients
    Matrix<T, Dynamic, Dynamic> Bin( ph + 1, ph2 + 1 );
    bezalfs.setZero();
    Bin.setZero();
    binomialCoef( Bin );
    // Compute Bezier degree elevation coefficients
    T inv;
    int mpi;
    bezalfs( 0, 0 ) = bezalfs( ph, p ) = T( 1 );
    for ( i = 1; i <= ph2; i++ )
    {
        inv = T( 1 ) / Bin( ph, i );
        mpi = std::min( p, i );
        for ( j = std::max( 0, i - t ); j <= mpi; j++ )
        {
            bezalfs( i, j ) = inv * Bin( p, j ) * Bin( t, i - j );
        }
    }

    for ( i = ph2 + 1; i < ph; i++ )
    {
        mpi = std::min( p, i );
        for ( j = std::max( 0, i - t ); j <= mpi; j++ )
            bezalfs( i, j ) = bezalfs( ph - i, p - j );
    }

    P.resize( cP.size() * t * 3 ); // Allocate more control points than necessary
    U.resize( cP.size() * t * 3 + ph + 1 );
    int mh = ph;
    int kind = ph + 1;
    T ua = U( 0 );
    T ub = U( 0 );
    int r = -1;
    int oldr;
    int a = p;
    int b = p + 1;
    int cind = 1;
    int rbz, lbz = 1;
    int mul, save, s;
    T alf;
    int first, last, kj;
    T den, bet, gam, numer;

    P[0] = cP[0];
    for ( i = 0; i <= ph; i++ )
    {
        U( i ) = ua;
    }

    // Initialize the first Bezier segment
    for ( i = 0; i <= p; i++ )
        bpts[i] = cP[i];
    while ( b < m )
    { // Big loop thru knot vector
        i = b;
        while ( b < m && cU( b ) == cU( b + 1 ) ) // for some odd reasons... == doesn't work
            b++;
        mul = b - i + 1;
        mh += mul + t;
        ub = cU( b );
        oldr = r;
        r = p - mul;
        if ( oldr > 0 )
            lbz = ( oldr + 2 ) / 2;
        else
            lbz = 1;
        if ( r > 0 )
            rbz = ph - ( r + 1 ) / 2;
        else
            rbz = ph;
        if ( r > 0 )
        { // Insert knot to get Bezier segment
            numer = ub - ua;
            for ( k = p; k > mul; k-- )
            {
                alphas[k - mul - 1] = numer / ( cU( a + k ) - ua );
            }
            for ( j = 1; j <= r; j++ )
            {
                save = r - j;
                s = mul + j;
                for ( k = p; k >= s; k-- )
                {
                    bpts[k] = alphas[k - s] * bpts[k] + ( T( 1 ) - alphas[k - s] ) * bpts[k - 1];
                }
                Nextbpts[save] = bpts[p];
            }
        }

        for ( i = lbz; i <= ph; i++ )
        { // Degree elevate Bezier,  only the points lbz,...,ph are used
            ebpts[i] = Matrix<T, N, 1>::Zero( N );
            mpi = std::min( p, i );
            for ( j = std::max( 0, i - t ); j <= mpi; j++ )
                ebpts[i] += bezalfs( i, j ) * bpts[j];
        }

        if ( oldr > 1 )
        { // Must remove knot u=c.U[a] oldr times
            // if(oldr>2) // Alphas on the right do not change
            //	alfj = (ua-U[kind-1])/(ub-U[kind-1]) ;
            first = kind - 2;
            last = kind;
            den = ub - ua;
            bet = ( ub - U( kind - 1 ) ) / den;
            for ( int tr = 1; tr < oldr; tr++ )
            { // Knot removal loop
                i = first;
                j = last;
                kj = j - kind + 1;
                while ( j - i > tr )
                { // Loop and compute the new control points for one removal step
                    if ( i < cind )
                    {
                        alf = ( ub - U( i ) ) / ( ua - U( i ) );
                        P[i] = alf * P[i] + ( T( 1 ) - alf ) * P[i - 1];
                    }
                    if ( j >= lbz )
                    {
                        if ( j - tr <= kind - ph + oldr )
                        {
                            gam = ( ub - U( j - tr ) ) / den;
                            ebpts[kj] = gam * ebpts[kj] + ( T( 1 ) - gam ) * ebpts[kj + 1];
                        }
                        else
                        {
                            ebpts[kj] = bet * ebpts[kj] + ( T( 1 ) - bet ) * ebpts[kj + 1];
                        }
                    }
                    ++i;
                    --j;
                    --kj;
                }
                --first;
                ++last;
            }
        }

        if ( a != p ) // load the knot u=c.U[a]
            for ( i = 0; i < ph - oldr; i++ )
            {
                U( kind++ ) = ua;
            }
        for ( j = lbz; j <= rbz; j++ )
        { // load control points onto the curve
            P[cind++] = ebpts[j];
        }

        if ( b < m )
        { // Set up for next pass thru loop
            for ( j = 0; j < r; j++ )
                bpts[j] = Nextbpts[j];
            for ( j = r; j <= p; j++ )
                bpts[j] = cP[b - p + j];
            a = b;
            b++;
            ua = ub;
        }
        else
        {
            for ( i = 0; i <= ph; i++ )
                U( kind + i ) = ub;
        }
    }
    P.resize( mh - ph ); // Resize to the proper number of control points
    U.resize( mh + 1 );
}

template <typename T, int N>
void knotInsertion( T u, int r, KnotVector<T>& U, ContPtsList<T, N>& P )
{
    int n = U.GetDOF();
    int p = U.GetDegree();
    auto cP = P;
    auto cU = U;
    int m = n + p;
    int nq = n + r;
    int k, s = 0;
    int i, j;
    k = U.FindSpan( u );
    P.resize( nq );
    U.resize( nq + p + 1 );
    for ( i = 0; i <= k; i++ )
        U( i ) = cU( i );
    for ( i = 1; i <= r; i++ )
        U( k + i ) = u;
    for ( i = k + 1; i <= m; i++ )
        U( i + r ) = cU( i );

    ContPtsList<T, N> R( p + 1 );
    for ( i = 0; i <= k - p; i++ )
        P[i] = cP[i];
    for ( i = k - s; i < n; i++ )
        P[i + r] = cP[i];
    for ( i = 0; i <= p - s; i++ )
        R[i] = cP[k - p + i];
    int L;
    T alpha;
    for ( j = 1; j <= r; j++ )
    {
        L = k - p + j;
        for ( i = 0; i <= p - j - s; i++ )
        {
            alpha = ( u - cU( L + i ) ) / ( cU( i + k + 1 ) - cU( L + i ) );
            R[i] = alpha * R[i + 1] + ( T( 1 ) - alpha ) * R[i];
        }
        P[L] = R[0];
        P[k + r - j - s] = R[p - j - s];
    }
    for ( i = L + 1; i < k - s; i++ )
        P[i] = R[i - L];
}

template <typename T, int N>
void refineKnotVectorCurve( const KnotVector<T>& X, KnotVector<T>& U, ContPtsList<T, N>& P )
{
    int n = U.GetDOF() - 1;
    int p = U.GetDegree();
    int m = n + p + 1;
    int a, b;
    int r = static_cast<int>( X.GetSize() - 1 );
    auto cP = P;
    auto cU = U;
    P.resize( r + n + 2 );
    U.resize( r + n + p + 3 );
    a = cU.FindSpan( X[0] );
    b = cU.FindSpan( X[r] );
    ++b;
    int j;
    for ( j = 0; j <= a - p; j++ )
        P[j] = cP[j];
    for ( j = b - 1; j <= n; j++ )
        P[j + r + 1] = cP[j];
    for ( j = 0; j <= a; j++ )
        U( j ) = cU( j );
    for ( j = b + p; j <= m; j++ )
        U( j + r + 1 ) = cU( j );
    int i = b + p - 1;
    int k = b + p + r;
    for ( j = r; j >= 0; j-- )
    {
        while ( X[j] <= cU[i] && i > a )
        {
            P[k - p - 1] = cP[i - p - 1];
            U( k ) = cU( i );
            --k;
            --i;
        }
        P[k - p - 1] = P[k - p];
        for ( int l = 1; l <= p; l++ )
        {
            int ind = k - p + l;
            T alpha = U[k + l] - X[j];
            if ( alpha == T( 0 ) )
                P[ind - 1] = P[ind];
            else
                alpha /= U( k + l ) - cU( i - p + l );
            P[ind - 1] = alpha * P[ind - 1] + ( T( 1 ) - alpha ) * P[ind];
        }
        U( k ) = X[j];
        --k;
    }
}

template <int N>
DifferentialPatternList_ptr PartialDerPattern( int r )
{
    std::vector<int> kk( r );
    DifferentialPatternList_ptr a( new DifferentialPatternList );
    std::function<void( int, int, std::vector<int>&, int, int, DifferentialPatternList_ptr& )> recursive;
    recursive = [&]( int D, int i, std::vector<int>& k, int n, int start, std::unique_ptr<std::vector<std::vector<int>>>& a ) {
        if ( n == i )
        {
            std::vector<int> m;
            int it = 0;
            for ( int it1 = 0; it1 < D; ++it1 )
            {
                int amount = 0;
                while ( find( k.begin(), k.end(), it ) != k.end() )
                {
                    amount++;
                    it++;
                }
                m.push_back( amount );
                it++;
            }
            a->push_back( m );
        }
        else
        {
            for ( int jj = start; jj < D + i - ( i - n ); ++jj )
            {
                k[n] = jj;
                recursive( D, i, k, n + 1, jj + 1, a );
            }
        }
    };
    recursive( N, r, kk, 0, 0, a );
    return a;
}

template <typename T>
std::unique_ptr<ExtractionOperatorContainer<T>> BezierExtraction( const KnotVector<T>& knot )
{
    std::unique_ptr<ExtractionOperatorContainer<T>> result( new ExtractionOperatorContainer<T> );
    int m = knot.GetSize();
    int p = knot.GetDegree();
    int a = p + 1;
    int b = a + 1;
    int nb = 1;
    std::vector<T> alphas( p + 1, 0 );
    ExtractionOperator<T> C = ExtractionOperator<T>::Identity( p + 1, p + 1 );
    while ( b < m )
    {
        ExtractionOperator<T> C_next = ExtractionOperator<T>::Identity( p + 1, p + 1 );
        int i = b;
        while ( b < m && knot[b] == knot[b - 1] )
        {
            b++;
        }
        int mult = b - i + 1;
        if ( mult < p )
        {
            T numer = knot[b - 1] - knot[a - 1];
            for ( int j = p; j > mult; j-- )
            {
                alphas[j - mult - 1] = numer / ( knot[a + j - 1] - knot[a - 1] );
            }
            int r = p - mult;
            for ( int j = 1; j <= r; j++ )
            {
                int save = r - j + 1;
                int s = mult + j;
                for ( int k = p + 1; k >= s + 1; k-- )
                {
                    T alpha = alphas[k - s - 1];
                    C.col( k - 1 ) = alpha * C.col( k - 1 ) + ( T( 1 ) - alpha ) * C.col( k - 2 );
                }

                if ( b < m )
                {
                    for ( int l = 0; l <= j; l++ )
                    {
                        C_next( save + l - 1, save - 1 ) = C( p - j + l, p );
                    }
                }
            }
        }

        nb++;
        if ( b < m )
        {
            result->push_back( C );
            C = C_next;
            a = b;
            b++;
        }
    }
    result->push_back( C );
    return result;
}

template <typename T>
std::unique_ptr<ExtractionOperatorContainer<T>> BezierReconstruction( const KnotVector<T>& knot )
{
    auto res = BezierExtraction<T>( knot );
    for ( auto& i : *res )
    {
        i = i.inverse();
    }
    return res;
}

int Binomial( const int n, const int k );

template <typename T>
Matrix<T, Dynamic, Dynamic> Gramian( int p )
{
    int n = p + 1;
    Matrix<T, Dynamic, Dynamic> res( n, n );
    res.setZero();
    for ( int i = 0; i < n; i++ )
    {
        for ( int j = 0; j <= i; j++ )
        {
            res( i, j ) = Binomial( p, i ) * Binomial( p, j ) / ( 2 * p + 1 ) / Binomial( 2 * p, i + j );
        }
    }
    res = res.template selfadjointView<Eigen::Lower>();
    return res;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> GramianInverse( int p )
{
    int n = p + 1;
    Matrix<T, Dynamic, Dynamic> res( n, n );
    res.setZero();
    for ( int i = 0; i < n; i++ )
    {
        for ( int j = 0; j <= i; j++ )
        {
            T sum = 0;
            for ( int k = 0; k <= std::min( i, j ); k++ )
            {
                sum += ( 2 * k + 1 ) * Binomial( p + k + 1, p - i ) * Binomial( p - k, p - i ) *
                       Binomial( p + k + 1, p - j ) * Binomial( p - k, p - j );
            }
            res( i, j ) = sum * pow( -1, i + j ) / Binomial( p, i ) / Binomial( p, j );
        }
    }
    res = res.template selfadjointView<Eigen::Lower>();
    return res;
}

template <typename T>
std::vector<T> AllBernstein( int p, T u )
{
    std::vector<T> res( p + 1 );
    res[0] = 1;
    T u1 = 1 - u;
    for ( int j = 1; j <= p; j++ )
    {
        T saved = 0;
        for ( int k = 0; k < j; k++ )
        {
            T temp = res[k];
            res[k] = saved + u1 * temp;
            saved = u * temp;
        }
        res[j] = saved;
    }
    return res;
}

template <int N, int d_from, int d_to, typename T>
bool MapParametricPoint( const PhyTensorBsplineBasis<d_from, N, T>* const from_domain,
                         const Eigen::Matrix<T, Eigen::Dynamic, 1>& from_point,
                         const PhyTensorBsplineBasis<d_to, N, T>* const to_domain,
                         Eigen::Matrix<T, Eigen::Dynamic, 1>& to_point )
{
    ASSERT( from_domain->InDomain( from_point ), "The point about to be mapped is out of the domain." );
    Eigen::Matrix<T, N, 1> physical_point = from_domain->AffineMap( from_point );
    bool res = to_domain->InversePts( physical_point, to_point );
    to_domain->FixOnBoundary( to_point );
    return res;
}

std::map<int, int> IndicesInverseMap( const std::vector<int>& forward_map );

std::vector<int> IndicesIntersection( const std::vector<int>& indices_a, const std::vector<int>& indices_b );

std::vector<int> NClosestDof( const std::vector<int>& dofs, int target_dof, int n );

std::vector<int> IndicesDifferentiation( const std::vector<int>& indices_a, const std::vector<int>& indices_b );

std::vector<int> IndicesUnion( const std::vector<int>& indices_a, const std::vector<int>& indices_b );

template <typename T>
std::set<int> ColIndicesSet( const std::vector<Eigen::Triplet<T>>& triplet )
{
    std::set<int> res;
    for ( const auto& i : triplet )
    {
        res.insert( i.col() );
    }
    return res;
}

template <typename T>
std::vector<int> ColIndicesVector( const std::vector<Eigen::Triplet<T>>& triplet )
{
    std::set<int> res;
    for ( const auto& i : triplet )
    {
        res.insert( i.col() );
    }
    std::vector<int> res_vector( res.begin(), res.end() );
    return res_vector;
}

template <typename T>
std::set<int> RowIndicesSet( const std::vector<Eigen::Triplet<T>>& triplet )
{
    std::set<int> res;
    for ( const auto& i : triplet )
    {
        res.insert( i.row() );
    }
    return res;
}

template <typename T>
std::vector<int> RowIndicesVector( const std::vector<Eigen::Triplet<T>>& triplet )
{
    std::set<int> res;
    for ( const auto& i : triplet )
    {
        res.insert( i.row() );
    }
    std::vector<int> res_vector( res.begin(), res.end() );
    return res_vector;
}
template <typename T>
void removeRow( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, unsigned int rowToRemove )
{
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if ( rowToRemove < numRows )
        matrix.block( rowToRemove, 0, numRows - rowToRemove, numCols ) =
            matrix.block( rowToRemove + 1, 0, numRows - rowToRemove, numCols );

    matrix.conservativeResize( numRows, numCols );
}

template <typename T>
void removeColumn( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, unsigned int colToRemove )
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if ( colToRemove < numCols )
        matrix.block( 0, colToRemove, numRows, numCols - colToRemove ) =
            matrix.block( 0, colToRemove + 1, numRows, numCols - colToRemove );

    matrix.conservativeResize( numRows, numCols );
}

template <typename T>
std::tuple<Eigen::Matrix<T, 3, 1>, Eigen::Matrix<T, 3, 1>, Eigen::Matrix<T, 3, 1>> CovariantToContravariant(
    const Eigen::Matrix<T, 3, 1>& v1, const Eigen::Matrix<T, 3, 1>& v2, const Eigen::Matrix<T, 3, 1>& v3 )
{
    T J = v1.dot( v2.cross( v3 ) );
    return std::make_tuple( 1.0 / J * v2.cross( v3 ), 1.0 / J * v3.cross( v1 ), 1.0 / J * v1.cross( v2 ) );
}

template <typename T>
std::pair<T, T> SinCosBetweenTwoUniVec( const Eigen::Matrix<T, 3, 1>& from,
                                        const Eigen::Matrix<T, 3, 1>& to,
                                        const Eigen::Matrix<T, 3, 1>& normal )
{
    T c = from.dot( to );
    T s = ( normal.cross( from ) ).dot( to );
    return std::make_pair( s, c );
}

template <typename T>
Eigen::Matrix<T, 3, 3> RotationMatrix( const Eigen::Matrix<T, 3, 1>& from, const Eigen::Matrix<T, 3, 1>& to )
{
    Eigen::Matrix<T, 3, 1> n_from = from.normalized();
    Eigen::Matrix<T, 3, 1> n_to = to.normalized();
    Eigen::Matrix<T, 3, 1> v = n_from.cross( n_to );
    auto sc = SinCosBetweenTwoUniVec( n_from, n_to, v );
    Eigen::Matrix<T, 3, 3> K;
    K << 0, -v( 2 ), v( 1 ), v( 2 ), 0, -v( 0 ), -v( 1 ), v( 0 ), 0;
    return Eigen::Matrix<T, 3, 3>::Identity() + sc.first * K + ( 1 - sc.second ) * K * K;
}

template <typename T>
Eigen::Matrix<T, 3, 3> RotationMatrix( const Eigen::Matrix<T, 3, 1>& n, const T s, const T c )
{
    Eigen::Matrix<T, 3, 3> K;
    K << 0, -n( 2 ), n( 1 ), n( 2 ), 0, -n( 0 ), -n( 1 ), n( 0 ), 0;
    return Eigen::Matrix<T, 3, 3>::Identity() + s * K + ( 1 - c ) * K * K;
}

template <typename T>
struct removeNoise_helper
{
    removeNoise_helper( const T& tol ) : m_tol( tol )
    {
    }

    inline const T operator()( const T& val ) const
    {
        return ( abs( val ) < m_tol ? 0 : val );
    }

    const T& m_tol;
};

template <typename T>
void removeNoise( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, const T tol )
{
    matrix.noalias() = matrix.unaryExpr( removeNoise_helper<T>( tol ) );
}

template <typename T>
Matrix<T, Dynamic, Dynamic> OrthonormalHelper( int size )
{
    static std::map<int, Matrix<T, Dynamic, Dynamic>> memo;
    auto it = memo.find( size );
    if ( it != memo.end() )
    {
        return it->second;
    }
    else
    {
        Matrix<T, Dynamic, 1> v = Matrix<T, Dynamic, 1>::Ones( size );
        JacobiSVD<Matrix<T, Dynamic, 1>> svd( v, ComputeFullU | ComputeFullV );
        memo[size] = svd.matrixU();
        return memo[size];
    }
}

template <typename ForwardIterator, typename T = typename std::iterator_traits<ForwardIterator>::value_type::Scalar>
SparseMatrix<T, ColMajor> SpVecToSpMat( ForwardIterator begin, ForwardIterator end )
{
    SparseMatrix<T, ColMajor> result;
    int num_of_cols = end - begin;
    int num_of_rows = begin->rows();
    result.resize( num_of_rows, num_of_cols );
    for ( auto it = begin; it != end; ++it )
    {
        int non_zeros = it->nonZeros();
        const auto inner_IndexPtr = it->innerIndexPtr();
        const auto valuePtr = it->valuePtr();
        for ( int i = 0; i < non_zeros; ++i )
        {
            result.coeffRef( *( inner_IndexPtr + i ), it - begin ) = *( valuePtr + i );
        }
    }
    return result;
}

// find a better way
template <typename ForwardIterator, typename T = typename std::iterator_traits<ForwardIterator>::value_type::second_type::Scalar>
SparseMatrix<T, ColMajor> SpVecPairToSpMat( ForwardIterator begin, ForwardIterator end )
{
    SparseMatrix<T, ColMajor> result;
    int num_of_cols = end - begin;
    int num_of_rows = begin->second.rows();
    result.resize( num_of_rows, num_of_cols );
    for ( auto it = begin; it != end; ++it )
    {
        int non_zeros = it->second.nonZeros();
        const auto inner_IndexPtr = it->second.innerIndexPtr();
        const auto valuePtr = it->second.valuePtr();
        for ( int i = 0; i < non_zeros; ++i )
        {
            result.coeffRef( *( inner_IndexPtr + i ), it - begin ) = *( valuePtr + i );
        }
    }
    return result;
}

template <typename Sp, typename T = typename Sp::Scalar>
std::vector<SparseVector<T, ColMajor>> OrthonormalSpVec( const Sp& sp )
{
    int size = sp.rows();
    int non_zeros = sp.nonZeros();
    if ( non_zeros == 1 )
    {
        return std::vector<SparseVector<T, ColMajor>>();
    }
    else
    {
        const auto inner_IndexPtr = sp.innerIndexPtr();
        std::vector<SparseVector<T, ColMajor>> result;
        Matrix<T, Dynamic, Dynamic> dense_orthonormal = OrthonormalHelper<T>( non_zeros );
        for ( int i = 1; i < non_zeros; ++i )
        {
            SparseVector<T, ColMajor> temp;
            temp.resize( size );
            for ( int j = 0; j < non_zeros; ++j )
            {
                temp.coeffRef( *( inner_IndexPtr + j ) ) = dense_orthonormal( j, i );
            }
            result.push_back( std::move( temp ) );
        }
        return result;
    }
}

template <typename SpPair, typename T = typename SpPair::second_type::Scalar>
std::vector<std::pair<int, SparseVector<T, ColMajor>>> OrthonormalSpVec( const SpPair& sp )
{
    int size = sp.second.rows();
    int non_zeros = sp.second.nonZeros();
    if ( non_zeros == 1 )
    {
        return std::vector<std::pair<int, SparseVector<T, ColMajor>>>{};
    }
    else
    {
        const auto inner_IndexPtr = sp.second.innerIndexPtr();
        std::vector<std::pair<int, SparseVector<T, ColMajor>>> result;
        Matrix<T, Dynamic, Dynamic> dense_orthonormal = OrthonormalHelper<T>( non_zeros );
        for ( int i = 1; i < non_zeros; ++i )
        {
            SparseVector<T, ColMajor> temp;
            temp.resize( size );
            for ( int j = 0; j < non_zeros; ++j )
            {
                temp.coeffRef( *( inner_IndexPtr + j ) ) = dense_orthonormal( j, i );
            }
            result.push_back( std::move( std::make_pair( sp.first, temp ) ) );
        }
        return result;
    }
}

template <class Input1, class Input2, class Output1, class Output2, class Output3>
void decompose_sets( Input1 first1, Input1 last1, Input2 first2, Input2 last2, Output1 result1, Output2 result2, Output3 result3 )
{
    while ( first1 != last1 && first2 != last2 )
    {
        if ( *first1 < *first2 )
        {
            *result1++ = *first1++;
        }
        else if ( *first2 < *first1 )
        {
            *result2++ = *first2++;
        }
        else
        {
            *result3++ = *first1++;
            ++first2; // skip common value in set2
        }
    }
    std::copy( first1, last1, result1 );
    std::copy( first2, last2, result2 );
}

// Assembly matrix A for polynomial completeness dual.
template <typename T>
std::vector<std::pair<int, Eigen::SparseVector<T>>> BasisAssemblyVecs( const KnotVector<T>& kv )
{
    // basic variables
    int dof = kv.GetDOF();
    int elements = kv.NumOfElements();
    int degree = kv.GetDegree();
    int dof_in_element = degree + 1;
    std::vector<std::pair<int, Eigen::SparseVector<T>>> result;
    for ( int i = 0; i < dof; ++i )
    {
        result.push_back( std::make_pair( i, Eigen::SparseVector<T>() ) );
        result[i].second.resize( elements * dof_in_element );
    }
    for ( int i = 0; i < elements; i++ )
    {
        auto indices_in_ele = kv.IndicesInElement( i );
        for ( int j = 0; j < dof_in_element; ++j )
        {
            result[indices_in_ele[j]].second.coeffRef( dof_in_element * i + j ) = 1;
        }
    }
    for ( auto& i : result )
    {
        i.second /= i.second.nonZeros();
    }
    return result;
}

int Factorial( const int n );

// inner product of Bernstein B^p and polynomials {1, t, ..., t^q}
template <typename T>
Matrix<T, Dynamic, Dynamic> BernsteinInnerPolynomial( const int p, const int q )
{
    Matrix<T, Dynamic, Dynamic> res( p + 1, q + 1 );
    for ( int i = 0; i <= p; i++ )
    {
        for ( int j = 0; j <= q; j++ )
        {
            res( i, j ) = Binomial( p, i ) * Factorial( i + j ) * Factorial( p - i ) / Factorial( 1 + j + p );
        }
    }
    return res;
}

// map {1, t,..., t^n} to {1, (t-a)/(b-a),..., (t-a)^n/(b-a)^n}
template <typename T>
Matrix<T, Dynamic, Dynamic> AffineMappingOp( const int p, const T a, const T b )
{
    Matrix<T, Dynamic, Dynamic> res( p + 1, p + 1 );
    res.setZero();
    for ( int j = 0; j <= p; j++ )
    {
        for ( int i = 0; i <= j; i++ )
        {
            res( i, j ) = Binomial( j, i ) * pow( 1.0 / ( b - a ), i ) * pow( -a / ( b - a ), j - i );
        }
    }
    return res;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> AffineMappingOp( const int p, const std::pair<T, T>& ab )
{
    const auto a = ab.first;
    const auto b = ab.second;
    return AffineMappingOp<T>( p, a, b );
}

template <typename T>
std::pair<T, T> AffineMappingCoef( const std::pair<T, T>& ab0, const std::pair<T, T>& ab1 )
{
    return std::make_pair( ( ab1.first - ab0.first ) / ( ab0.second - ab0.first ),
                           ( ab1.second - ab0.first ) / ( ab0.second - ab0.first ) );
}
} // namespace Accessory
