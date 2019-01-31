//
// Created by di miao on 10/17/17.
//

#pragma once

#include "QuadratureRule.h"
#include "Topology.hpp"
#include "Utility.hpp"
#include "Visitor.hpp"
#include <mutex>
#include <thread>

template <typename T>
struct MatrixData
{
    std::unique_ptr<std::vector<int>> _rowIndices;
    std::unique_ptr<std::vector<int>> _colIndices;
    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _matrix;

    MatrixData()
        : _rowIndices{std::make_unique<std::vector<int>>()},
          _colIndices{std::make_unique<std::vector<int>>()},
          _matrix{std::make_unique<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>()}
    {
    }

    MatrixData( const MatrixData& matrix_data )
    {
        _rowIndices = std::unique_ptr<std::vector<int>>( new std::vector<int>() );
        _colIndices = std::unique_ptr<std::vector<int>>( new std::vector<int>() );
        _matrix = std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
            new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>() );
        *_rowIndices = *( matrix_data._rowIndices );
        *_colIndices = *( matrix_data._colIndices );
        *_matrix = *( matrix_data._matrix );
    }

    MatrixData( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, std::vector<int>& row, std::vector<int>& col )
        : MatrixData()
    {
        *_matrix = std::move( matrix );
        *_rowIndices = std::move( row );
        *_colIndices = std::move( col );
        ASSERT( Check(), "Given data does not match for creating MatrixData.\n" );
    }

    MatrixData operator*( const MatrixData& matrix )
    {
        ASSERT( *( this->_colIndices ) == *( matrix._rowIndices ), "MatrixData multiply can't be performed.\n" );
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp = *( this->_matrix ) * ( *( matrix._matrix ) );
        auto col_indices = *( matrix._colIndices );
        auto row_indices = *( this->_rowIndices );
        return MatrixData( temp, row_indices, col_indices );
    }

    void RowRemove( const int& row_index )
    {
        ASSERT( row_index >= 0 && row_index < _rowIndices->size(), "Row index is out of range.\n" );
        _rowIndices->erase( _rowIndices->begin() + row_index );
        Accessory::removeRow( *_matrix, row_index );
    }

    int rows() const
    {
        return _matrix->rows();
    }

    int cols() const
    {
        return _matrix->cols();
    }

    MatrixData operator+( const MatrixData& matrix )
    {
        std::vector<int> sum_row, sum_col;
        std::set_union( _rowIndices->begin(), _rowIndices->end(), matrix._rowIndices->begin(),
                        matrix._rowIndices->end(), std::back_inserter( sum_row ) );
        std::set_union( _colIndices->begin(), _colIndices->end(), matrix._colIndices->begin(),
                        matrix._colIndices->end(), std::back_inserter( sum_col ) );
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp( sum_row.size(), sum_col.size() );
        temp.setZero();
        for ( int i = 0; i < sum_row.size(); ++i )
        {
            for ( int j = 0; j < sum_col.size(); ++j )
            {
                auto it_row = std::find( _rowIndices->begin(), _rowIndices->end(), sum_row[i] );
                auto it_col = std::find( _colIndices->begin(), _colIndices->end(), sum_col[j] );
                if ( it_row != _rowIndices->end() && it_col != _colIndices->end() )
                {
                    int ii = it_row - _rowIndices->begin();
                    int jj = it_col - _colIndices->begin();
                    temp( i, j ) += ( *_matrix )( ii, jj );
                }
            }
        }
        for ( int i = 0; i < sum_row.size(); ++i )
        {
            for ( int j = 0; j < sum_col.size(); ++j )
            {
                auto it_row = std::find( matrix._rowIndices->begin(), matrix._rowIndices->end(), sum_row[i] );
                auto it_col = std::find( matrix._colIndices->begin(), matrix._colIndices->end(), sum_col[j] );
                if ( it_row != matrix._rowIndices->end() && it_col != matrix._colIndices->end() )
                {
                    int ii = it_row - matrix._rowIndices->begin();
                    int jj = it_col - matrix._colIndices->begin();
                    temp( i, j ) += ( *matrix._matrix )( ii, jj );
                }
            }
        }
        return MatrixData( temp, sum_row, sum_col );
    }

    MatrixData& operator+=( const MatrixData& matrix )
    {
        MatrixData tmp = *this;
        *this = tmp + matrix;
        return *this;
    }

    MatrixData& operator=( const MatrixData& matrix_data )
    {
        // self-assignment guard
        if ( this == &matrix_data )
            return *this;

        // do the copy
        *_rowIndices = *( matrix_data._rowIndices );
        *_colIndices = *( matrix_data._colIndices );
        *_matrix = *( matrix_data._matrix );

        // return the existing object so we can chain this operator
        return *this;
    }

    std::vector<Eigen::Triplet<T>> ToTriplets()
    {
        std::vector<Eigen::Triplet<T>> res;
        for ( int i = 0; i < _matrix->rows(); i++ )
        {
            for ( int j = 0; j < _matrix->cols(); j++ )
            {
                res.push_back( Eigen::Triplet<T>( ( *_rowIndices )[i], ( *_colIndices )[j], ( *_matrix )( i, j ) ) );
            }
        }
        return res;
    }

    void Print() const
    {
        std::cout << "Row indices: ";
        for ( const auto& i : *_rowIndices )
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "Col indices: ";
        for ( const auto& i : *_colIndices )
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "Matrix: \n";
        std::cout << *_matrix;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    bool Check() const
    {
        if ( _rowIndices->size() == _matrix->rows() && _colIndices->size() == _matrix->cols() )
            return true;
        return false;
    }
};

template <typename T>
struct VectorData
{
    std::unique_ptr<std::vector<int>> _rowIndices;
    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1>> _vector;

    VectorData()
        : _rowIndices{std::make_unique<std::vector<int>>()}, _vector{std::make_unique<Eigen::Matrix<T, Eigen::Dynamic, 1>>()}
    {
    }
    VectorData( Eigen::Matrix<T, Eigen::Dynamic, 1>& vector, std::vector<int>& row ) : VectorData()
    {
        *_vector = std::move( vector );
        *_rowIndices = std::move( row );
        Check();
    }

    void Print() const
    {
        std::cout << "Row indices: ";
        for ( const auto& i : *_rowIndices )
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "Vector: \n";
        std::cout << std::setprecision( 16 ) << *_vector;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    bool Check() const
    {
        if ( _rowIndices->size() == _vector->rows() )
            return true;
        return false;
    }
};

template <int d, int N, typename T>
class DomainVisitor : public Visitor<d, N, T>
{
public:
    using Knot = typename QuadratureRule<T>::Coordinate;
    using Quadrature = typename QuadratureRule<T>::Quadrature;
    using QuadList = typename QuadratureRule<T>::QuadList;
    using KnotSpan = std::pair<Knot, Knot>;
    using KnotSpanlist = std::vector<KnotSpan>;
    using LoadFunctor = std::function<std::vector<T>( const Knot& )>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    DomainVisitor()
    {
    }

    //    Multi thread domain visitor
    void Visit( Element<d, N, T>* g )
    {
        QuadratureRule<T> quad_rule;
        KnotSpanlist knot_spans;
        Initialize( g );
        InitializeQuadratureRule( g, quad_rule );
        InitializeKnotSpans( g, knot_spans );
        std::vector<std::thread> threads( _num_of_threads );
        const int grainsize = knot_spans.size() / _num_of_threads;
        auto work_iter = knot_spans.begin();
        auto lambda = [&]( typename KnotSpanlist::iterator begin, typename KnotSpanlist::iterator end ) -> void {
            for ( auto i = begin; i != end; ++i )
            {
                LocalAssemble( g, quad_rule, *i );
            }
        };
        for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
        {
            *it = std::thread( lambda, work_iter, work_iter + grainsize );
            work_iter += grainsize;
        }
        threads.back() = std::thread( lambda, work_iter, knot_spans.end() );
        for ( auto& i : threads )
        {
            i.join();
        }
    }

protected:
    virtual void Initialize( Element<d, N, T>* g ){};

    //    Initialize quadrature rule
    virtual void InitializeQuadratureRule( Element<d, N, T>* g, QuadratureRule<T>& quad_rule )
    {
        if ( d == 0 )
        {
            quad_rule.SetUpQuadrature( 1 );
        }
        else
        {
            auto domain = g->GetDomain();
            quad_rule.SetUpQuadrature( domain->MaxDegree() + 1 );
        }
    }

    //    Initialize knot spans
    virtual void InitializeKnotSpans( Element<d, N, T>* g, KnotSpanlist& knot_spans )
    {
        g->GetDomain()->KnotSpanGetter( knot_spans );
    }

    //    Pure virtual method local assemble algorithm is needed to be implemented
    //    here
    virtual void LocalAssemble( Element<d, N, T>*, const QuadratureRule<T>&, const KnotSpan& ) = 0;

    virtual MatrixData<T> LocalStiffness( const std::vector<Matrix>& weight_basis,
                                          std::vector<int>& weight_basis_indices,
                                          const std::vector<Matrix>& basis,
                                          std::vector<int>& basis_indices,
                                          const std::vector<T>& quadrature_wegiht )
    {
        Matrix tmp( weight_basis[0].cols(), basis[0].cols() );
        tmp.setZero();
        for ( int i = 0; i < quadrature_wegiht.size(); ++i )
        {
            tmp += weight_basis[i].transpose() * quadrature_wegiht[i] * basis[i];
        }

        MatrixData<T> res;
        *( res._rowIndices ) = std::move( weight_basis_indices );
        *( res._colIndices ) = std::move( basis_indices );
        *( res._matrix ) = std::move( tmp );
        ASSERT( res.Check(), "Error in construct stiffness.\n" );
        return res;
    }

    virtual VectorData<T> LocalRhs( const std::vector<Matrix>& weight_basis,
                                    const std::vector<int>& weight_basis_indices,
                                    const std::vector<Matrix>& function_value,
                                    const std::vector<T>& quadrature_wegiht )
    {
        Vector tmp( weight_basis[0].cols() );
        tmp.setZero();
        for ( int i = 0; i < quadrature_wegiht.size(); ++i )
        {
            tmp += weight_basis[i].transpose() * quadrature_wegiht[i] * function_value[i];
        }
        VectorData<T> res;
        *( res._rowIndices ) = std::move( weight_basis_indices );
        *( res._vector ) = std::move( tmp );
        ASSERT( res.Check(), "Error in construct rhs.\n" );
        return res;
    }

    void CondensedTripletVia( const std::map<int, int>& row_map,
                              const std::map<int, int>& col_map,
                              const std::vector<Eigen::Triplet<T>>& original_triplet,
                              std::vector<Eigen::Triplet<T>>& mapped_triplet ) const
    {
        mapped_triplet.clear();
        for ( const auto& i : original_triplet )
        {
            auto it_row = row_map.find( i.row() );
            auto it_col = col_map.find( i.col() );
            if ( it_row != row_map.end() && it_col != col_map.end() )
            {
                mapped_triplet.push_back( Eigen::Triplet<T>( it_row->second, it_col->second, i.value() ) );
            }
        }
    }

    void CondensedTripletVia( const std::map<int, int>& row_map,
                              const std::vector<Eigen::Triplet<T>>& original_triplet,
                              std::vector<Eigen::Triplet<T>>& mapped_triplet ) const
    {
        mapped_triplet.clear();
        for ( const auto& i : original_triplet )
        {
            auto it_row = row_map.find( i.row() );
            if ( it_row != row_map.end() )
            {
                mapped_triplet.push_back( Eigen::Triplet<T>( it_row->second, 0, i.value() ) );
            }
        }
    }
    //    Convert non-zero Symmetric MatrixData elements to Triplet
    void SymmetricTriplet( const MatrixData<T>& matrix, std::vector<Eigen::Triplet<T>>& triplet, const T& tol = 1e-15 ) const
    {
        ASSERT( matrix._rowIndices->size() == matrix._colIndices->size(),
                "Given matrix data does not fit to symmetric assembler." );
        for ( int i = 0; i < matrix._rowIndices->size(); ++i )
        {
            for ( int j = i; j < matrix._colIndices->size(); ++j )
            {
                T tmp{( *matrix._matrix )( i, j )};
                if ( abs( tmp ) > tol )
                {
                    triplet.emplace_back( Eigen::Triplet<T>( ( *matrix._rowIndices )[i], ( *matrix._colIndices )[j], tmp ) );
                }
            }
        }
    }

    //    Convert non-zero MatrixData elements to Triplet
    void Triplet( const MatrixData<T>& matrix, std::vector<Eigen::Triplet<T>>& triplet, const T& tol = 1e-15 ) const
    {
        for ( int i = 0; i < matrix._rowIndices->size(); ++i )
        {
            for ( int j = 0; j < matrix._colIndices->size(); ++j )
            {
                T tmp{( *matrix._matrix )( i, j )};
                if ( abs( tmp ) > tol )
                {
                    triplet.emplace_back( Eigen::Triplet<T>( ( *matrix._rowIndices )[i], ( *matrix._colIndices )[j], tmp ) );
                }
            }
        }
    }
    //    Convert non-zero VectorData elements to Triplet
    void Triplet( const VectorData<T>& vector, std::vector<Eigen::Triplet<T>>& triplet, const T& tol = 1e-15 ) const
    {
        for ( int i = 0; i < vector._rowIndices->size(); ++i )
        {
            T tmp{( *vector._vector )( i )};
            if ( abs( tmp ) > tol )
            {
                triplet.emplace_back( Eigen::Triplet<T>( ( *vector._rowIndices )[i], 0, tmp ) );
            }
        }
    }

    MatrixData<T> ToMatrixData( const std::vector<Eigen::Triplet<T>>& triplet )
    {
        std::vector<int> col_indices = Accessory::ColIndicesVector( triplet );
        std::vector<int> row_indices = Accessory::RowIndicesVector( triplet );
        auto col_inverse_indices = Accessory::IndicesInverseMap( col_indices );
        auto row_inverse_indices = Accessory::IndicesInverseMap( row_indices );
        std::vector<Eigen::Triplet<T>> condensed_triplet;
        CondensedTripletVia( row_inverse_indices, col_inverse_indices, triplet, condensed_triplet );
        Eigen::SparseMatrix<T> tmp;
        tmp.resize( row_indices.size(), col_indices.size() );
        tmp.setFromTriplets( condensed_triplet.begin(), condensed_triplet.end() );
        Matrix matrix = Matrix( tmp );
        return MatrixData<T>( matrix, row_indices, col_indices );
    }

    bool IndexModifier( const std::map<int, int>& index_map, int& index ) const
    {
        auto it = index_map.find( index );
        if ( it != index_map.end() )
        {
            index = it->second;
            return true;
        }
        return false;
    }

    void MatrixDataIndexModifier( const std::map<int, int>& index_map, MatrixData<T>& matrix_data )
    {
        //        Row operation
        for ( auto it = matrix_data._rowIndices->begin(); it != matrix_data._rowIndices->end(); )
        {
            if ( !IndexModifier( index_map, *it ) )
            {
                it = matrix_data._rowIndices->erase( it );
                int row_num = it - matrix_data._rowIndices->begin();
                Accessory::removeRow( *matrix_data._matrix, row_num );
            }
            else
            {
                ++it;
            }
        }
        //        Column operation
        for ( auto it = matrix_data._colIndices->begin(); it != matrix_data._colIndices->end(); )
        {
            if ( !IndexModifier( index_map, *it ) )
            {
                it = matrix_data._colIndices->erase( it );
                int col_num = it - matrix_data._colIndices->begin();
                Accessory::removeColumn( *matrix_data._matrix, col_num );
            }
            else
            {
                ++it;
            }
        }
    }

    void VectorDataIndexModifier( const std::map<int, int>& index_map, VectorData<T>& vector_data )
    {
        //        Row operation
        for ( auto it = vector_data._rowIndices->begin(); it != vector_data._rowIndices->end(); )
        {
            if ( !IndexModifier( index_map, *it ) )
            {
                it = vector_data._rowIndices->erase( it );
                int row_num = it - vector_data._rowIndices->begin();
                Accessory::removeRow( *vector_data._vector, row_num );
            }
            else
            {
                ++it;
            }
        }
    }

    void MatrixAssembler( const int& row_dof,
                          const int& col_dof,
                          const std::vector<Eigen::Triplet<T>>& triplet,
                          Eigen::SparseMatrix<T>& matrix ) const
    {
        matrix.resize( row_dof, col_dof );
        matrix.setFromTriplets( triplet.cbegin(), triplet.cend() );
    }

    void MatrixAssembler( const int& row_dof, const int& col_dof, const std::vector<Eigen::Triplet<T>>& triplet, Matrix& matrix ) const
    {
        Eigen::SparseMatrix<T> sparse_matrix;
        MatrixAssembler( row_dof, col_dof, triplet, sparse_matrix );
        matrix = Matrix( sparse_matrix );
    }

    void VectorAssembler( const int& row_dof, const std::vector<Eigen::Triplet<T>>& triplet, Eigen::SparseMatrix<T>& vector ) const
    {
        vector.resize( row_dof, 1 );
        vector.setFromTriplets( triplet.cbegin(), triplet.cend() );
    }

    Matrix Solve( const Eigen::SparseMatrix<T>& gramian, const Eigen::SparseMatrix<T>& rhs ) const
    {
        ASSERT( gramian.rows() == gramian.cols(), "The size of given gramian matrix is not correct.\n" );
        Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower | Eigen::Upper> cg;
        cg.setMaxIterations( 10 * gramian.rows() );
        cg.compute( gramian );
        Matrix res = cg.solve( rhs );
        return res;
    }

    Matrix SolveLU( const Eigen::SparseMatrix<T>& gramian, const Eigen::SparseMatrix<T>& rhs ) const
    {
        using namespace Eigen;
        ASSERT( gramian.rows() == gramian.cols(), "The size of given gramian matrix is not correct.\n" );
        SparseLU<SparseMatrix<T>> solver;

        // Compute the ordering permutation vector from the structural pattern of A
        solver.analyzePattern( gramian );
        // Compute the numerical factorization
        solver.factorize( gramian );
        // Use the factors to solve the linear system
        Matrix res = solver.solve( rhs );
        return res;
    }

    Matrix Solve( const Matrix& gramian, const Matrix& rhs ) const
    {
        ASSERT( gramian.rows() == gramian.cols(), "The size of given gramian matrix is not correct.\n" );
        Eigen::ConjugateGradient<Matrix, Eigen::Lower | Eigen::Upper> cg;
        cg.compute( gramian );
        Matrix res = cg.solve( rhs );
        return res;
    }

    Matrix SolveNonSymmetric( const Matrix& gramian, const Matrix& rhs ) const
    {
        ASSERT( gramian.rows() == gramian.cols(), "The size of given gramian matrix is not correct.\n" );
        return gramian.partialPivLu().solve( rhs );
    }

    inline void ThreadSetter( const unsigned int& threads )
    {
        _num_of_threads = threads;
    }

protected:
    std::mutex _mutex;
    unsigned int _num_of_threads{std::min( std::thread::hardware_concurrency(), 12 )};
    // u_int _num_of_threads{1};
};
