//
// Created by miaodi on 25/12/2016.
//

#pragma once

#include "KnotVector.h"
#include "QuadratureRule.h"
#include "Utility.hpp"
#include <memory>

template <typename T>
class DualBasis;

template <typename T>
class BsplineBasis
{
public:
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using block = Eigen::Block<matrix>;

    typedef std::pair<int, T> BasisFunVal;
    typedef std::vector<BasisFunVal> BasisFunValPac;
    typedef std::unique_ptr<BasisFunValPac> BasisFunValPac_ptr;
    using BasisFunValDerAll = std::pair<int, std::vector<T>>;
    using BasisFunValDerAllList = std::vector<BasisFunValDerAll>;
    using BasisFunValDerAllList_ptr = std::unique_ptr<BasisFunValDerAllList>;

    BsplineBasis();

    BsplineBasis( KnotVector<T> target );

    int GetDegree() const;

    int GetDof() const;

    int FindSpan( const T& u ) const;

    BasisFunValDerAllList_ptr EvalDerAll( const T& u, int i ) const;

    BasisFunValPac_ptr Eval( const T& u, const int i = 0 ) const;

    T EvalSingle( const T& u, const int n, const int i = 0 ) const;

    vector Support( const int i ) const;

    vector InSpan( const T& u ) const;

    inline T DomainStart() const
    {
        return _basisKnot[GetDegree()];
    }

    inline T DomainEnd() const
    {
        return _basisKnot[GetDof()];
    }

    int NumActive() const
    {
        return GetDegree() + 1;
    }

    inline const KnotVector<T>& Knots() const
    {
        return _basisKnot;
    }

    inline bool InDomain( T const& u ) const
    {
        return ( ( u >= DomainStart() ) && ( u <= DomainEnd() ) );
    }

    inline void PrintKnots() const
    {
        _basisKnot.printKnotVector();
    }

    inline void PrintUniKnots() const
    {
        _basisKnot.printUnique();
    }

    inline int FirstActive( T u ) const
    {
        return ( InDomain( u ) ? FindSpan( u ) - GetDegree() : 0 );
    }

    std::unique_ptr<matrix> BasisWeight() const;

    // Assembly matrix A for polynomial completeness dual.
    std::vector<Eigen::SparseVector<T>> BasisAssemblyVecs() const;

    // lhs and rhs assembler for polynomial completeness dual.
    std::pair<matrix, matrix> LhsRhsAssembler( int num_of_completeness ) const;

    bool IsActive( const int i, const T u ) const;

    void BezierDualInitialize();

    // Reduce the order of first two and last two elements by one (Serve as the Lagrange multiplier). The weights for boundary basis are computed. (Only C^{p-1} spline are considered.)
    void ModifyBoundaryInitialize();

    // Return the evaluation of the modified b-spline basis functions.
    BasisFunValDerAllList_ptr EvalModifiedDerAll( const T& u, int i ) const;

    BasisFunValDerAllList_ptr BezierDual( const T& u ) const;

    // Return the evaluation of the modified b-spline basis functions.
    BasisFunValDerAllList_ptr EvalCodimensionBezierDual( const T& u ) const;

    const matrix& AssemblerGetter() const
    {
        return _dualBasis._spAssemble;
    }

protected:
    KnotVector<T> _basisKnot;
    std::vector<matrix> _reconstruction;
    mutable matrix _basisWeight;
    matrix _gramianInv;
    DualBasis<T> _dualBasis;
    std::vector<std::pair<int, Eigen::Ref<matrix>>> _localWeightContainer;
    bool _complete_dual{true};
};

template <typename T>
class DualBasis
{
public:
    using ExtractionOperator = Accessory::ExtractionOperator<T>;
    using ExtractionOperatorContainer = Accessory::ExtractionOperatorContainer<T>;

public:
    DualBasis() : _basisKnot{nullptr}, _codimension{0}
    {
    }
    DualBasis( KnotVector<T>* kv, int cd ) : _basisKnot{kv}, _codimension{cd}
    {
        Initialization();
    }

    std::pair<T, T> Support( const int i ) const
    {
        const int deg = _basisKnot->GetDegree();
        ASSERT( i < _basisKnot->GetDOF(), "Invalid index of basis function." );
        return std::make_pair( ( *_basisKnot )[i], ( *_basisKnot )[i + deg + 1] );
    }

    void Initialization()
    {
        const int deg = _basisKnot->GetDegree();
        const int dof = _basisKnot->GetDOF();

        _gramian = Accessory::Gramian<T>( deg );
        _extractionOp = *Accessory::BezierExtraction<T>( *_basisKnot );

        // first step, get all assembly vectors for each basis function
        auto assembly_vectors = Accessory::BasisAssemblyVecs<T>( *_basisKnot );
        std::vector<std::pair<int, Eigen::SparseVector<T>>> assembly_null_vectors;

        // completeness of constructed dual basis functions
        // const int polynomial_completeness = _basisKnot->GetDegree() - 1;
        const int polynomial_completeness = 0;

        // move some assembly vectors to null space due to the codimension
        std::move( assembly_vectors.begin(), assembly_vectors.begin() + _codimension, std::back_inserter( assembly_null_vectors ) );
        std::move( assembly_vectors.rbegin(), assembly_vectors.rbegin() + _codimension, std::back_inserter( assembly_null_vectors ) );

        for ( auto& i : assembly_null_vectors )
        {
            i.second = i.second * ( i.second.nonZeros() ) / sqrt( ( i.second.nonZeros() ) );
        }
        ASSERT( assembly_vectors.size() > polynomial_completeness,
                "knot vector does not satisfy the minimum requirement for constructing complete dual basis.\n" );

        auto assembly_null_vectors_temp = assembly_null_vectors;

        for ( const auto& i : assembly_null_vectors_temp )
        {
            auto res = Accessory::OrthonormalSpVec( i );
            std::move( res.begin(), res.end(), std::back_inserter( assembly_null_vectors ) );
        }
        for ( auto it = assembly_vectors.begin() + _codimension; it != assembly_vectors.end() - _codimension; ++it )
        {
            auto res = Accessory::OrthonormalSpVec( *it );
            std::move( res.begin(), res.end(), std::back_inserter( assembly_null_vectors ) );
        }

        // obtain local gramian matrix
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bernstein_inner_polynomial =
            Accessory::BernsteinInnerPolynomial<T>( deg, polynomial_completeness );

        // obtain the initial guess of matrix W
        _spAssemble = Accessory::SpVecPairToSpMat( assembly_vectors.begin() + _codimension, assembly_vectors.end() - _codimension );

        // set up all dimensions exclude
        std::vector<int> indices( dof - 2 * _codimension );
        std::iota( indices.begin(), indices.end(), _codimension );

        // construct element spans
        const auto spans = _basisKnot->KnotSpans();

        for ( const auto& i : assembly_null_vectors )
        {
            auto closest_dofs = Accessory::NClosestDof( indices, i.first, polynomial_completeness + 1 );

            // find support of activated dofs.
            std::pair<T, T> span = std::make_pair( std::numeric_limits<T>::max(), std::numeric_limits<T>::min() );

            for ( const auto each_dof : closest_dofs )
            {
                auto support = Support( each_dof );
                span.first = std::min( span.first, support.first );
                span.second = std::max( span.second, support.second );
            }

            const auto involved_elements = _basisKnot->SpanToElements( span );

            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> discontinuous_spline_inner_local_poly(
                involved_elements.size() * ( deg + 1 ), polynomial_completeness + 1 );
            discontinuous_spline_inner_local_poly.setZero();
            for ( int j = 0; j < involved_elements.size(); ++j )
            {
                Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> local_mat =
                    discontinuous_spline_inner_local_poly.block( ( deg + 1 ) * j, 0, deg + 1, polynomial_completeness + 1 );

                const auto ab = Accessory::AffineMappingCoef<T>( spans[involved_elements[j]], span );

                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mapping_op =
                    Accessory::AffineMappingOp<T>( polynomial_completeness, ab );

                const T jacobian = ( spans[involved_elements[j]].second - spans[involved_elements[j]].first ) /
                                   ( *( _basisKnot->end() - 1 ) - *( _basisKnot->begin() ) );

                local_mat = _extractionOp[involved_elements[j]] * bernstein_inner_polynomial * mapping_op * jacobian;
            }

            // assemble lhs
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lhs( polynomial_completeness + 1, polynomial_completeness + 1 );

            for ( int j = 0; j < closest_dofs.size(); ++j )
            {
                const auto& assembly_vec = assembly_vectors[closest_dofs[j]].second;
                lhs.block( j, 0, 1, polynomial_completeness + 1 ) =
                    assembly_vec
                        .block( ( deg + 1 ) * involved_elements[0], 0, ( deg + 1 ) * involved_elements.size(), 1 )
                        .transpose() *
                    assembly_vec.nonZeros() * discontinuous_spline_inner_local_poly;
            }
            // assemble rhs
            Eigen::Matrix<T, 1, Eigen::Dynamic> rhs( 1, polynomial_completeness + 1 );
            rhs = i.second.block( ( deg + 1 ) * involved_elements[0], 0, ( deg + 1 ) * involved_elements.size(), 1 ).transpose() *
                  discontinuous_spline_inner_local_poly;

            // solve
            Eigen::Matrix<T, Eigen::Dynamic, 1> local_weight = lhs.transpose().fullPivLu().solve( rhs.transpose() );

            // assemble to Matrix W
            for ( int j = 0; j < closest_dofs.size(); ++j )
            {
                _spAssemble.col( closest_dofs[j] - _codimension ) += *( local_weight.data() + j ) * i.second;
            }
        }

        for ( int i = 0; i < spans.size(); ++i )
        {
            int start_dof, end_dof;
            const auto indices = _basisKnot->IndicesInElement( i );
            start_dof = indices[0] < _codimension ? indices[0] : ( indices[0] - _codimension );
            end_dof = start_dof;
            while ( start_dof - 1 >= 0 && _spAssemble.block( ( deg + 1 ) * i, start_dof - 1, deg + 1, 1 ).norm() > 0 )
            {
                start_dof--;
            }
            while ( end_dof < _spAssemble.cols() && _spAssemble.block( ( deg + 1 ) * i, end_dof, deg + 1, 1 ).norm() > 0 )
            {
                end_dof++;
            }
            _localWeightContainer.emplace_back( std::make_pair(
                start_dof, _spAssemble.block( ( deg + 1 ) * i, start_dof, deg + 1, end_dof - start_dof ) ) );
        }
        // std::cout << _spAssemble << std::endl;
    }

public:
    KnotVector<T>* _basisKnot;
    int _codimension;
    ExtractionOperatorContainer _extractionOp;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _gramian;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _spAssemble;
    std::vector<std::pair<int, Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>> _localWeightContainer;
};