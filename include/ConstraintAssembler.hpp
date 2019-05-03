#pragma once
#include "BiharmonicInterfaceH1.hpp"
#include "BiharmonicInterfaceVisitor.hpp"
#include "DofMapper.hpp"
#include "PoissonInterfaceVisitor.hpp"
#include "Surface.hpp"
#include <map>
#include <numeric>

template <int d, int N, typename T>
class ConstraintAssembler
{
public:
    ConstraintAssembler( DofMapper& dof ) : _dof( dof )
    {
    }

    void ConstraintCreator( const std::vector<std::shared_ptr<Surface<N, T>>>& cells )
    {
        _matrix_data_container.clear();
        _vertex_indices.clear();
        _involved_indices.clear();
        _additional_constraint.clear();
        for ( auto& i : cells )
        {
            for ( int j = 0; j < 4; j++ )
            {
                if ( i->EdgePointerGetter( j )->IsMatched() && i->EdgePointerGetter( j )->IsSlave() )
                {
                    BiharmonicInterfaceVisitor<N, T> biharmonic_interface;
                    i->EdgePointerGetter( j )->Accept( biharmonic_interface );
                    int slave_id = biharmonic_interface.SlaveID();
                    int master_id = biharmonic_interface.MasterID();
                    int slave_starting_dof = _dof.StartingDof( slave_id );
                    int master_starting_dof = _dof.StartingDof( master_id );
                    for ( int k = 0; k <= 1; k++ )
                    {
                        auto vertex = i->EdgePointerGetter( j )->VertexPointerGetter( k );
                        auto slave_vert_ind = i->EdgePointerGetter( j )->VertexPointerGetter( k )->Indices( 1, 1 );
                        std::for_each( slave_vert_ind.begin(), slave_vert_ind.end(),
                                       [&]( int& index ) { index += slave_starting_dof; } );
                        _vertex_indices.insert( _vertex_indices.end(), slave_vert_ind.begin(), slave_vert_ind.end() );
                    }
                    auto constraint_data = biharmonic_interface.ConstraintData();
                    std::for_each( constraint_data._rowIndices->begin(), constraint_data._rowIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( constraint_data._colIndices->begin(), constraint_data._colIndices->end(),
                                   [&]( int& index ) { index += master_starting_dof; } );
                    _matrix_data_container.push_back( std::move( constraint_data ) );
                }
            }
        }

        std::sort( _vertex_indices.begin(), _vertex_indices.end() );
        _vertex_indices.erase( unique( _vertex_indices.begin(), _vertex_indices.end() ), _vertex_indices.end() );
        for ( const auto& i : _matrix_data_container )
        {
            _involved_indices.insert( _involved_indices.end(), i._rowIndices->begin(), i._rowIndices->end() );
            _involved_indices.insert( _involved_indices.end(), i._colIndices->begin(), i._colIndices->end() );
        }
        std::sort( _involved_indices.begin(), _involved_indices.end() );
        _involved_indices.erase( unique( _involved_indices.begin(), _involved_indices.end() ), _involved_indices.end() );
    }

    void ConstraintCodimensionCreator( const std::vector<std::shared_ptr<Surface<N, T>>>& cells )
    {
        _matrix_data_container.clear();
        _vertex_indices.clear();
        _involved_indices.clear();
        _additional_constraint.clear();
        int total_dof = _dof.TotalDof();
        for ( auto& i : cells )
        {
            for ( int j = 0; j < 4; j++ )
            {
                if ( i->EdgePointerGetter( j )->IsMatched() && i->EdgePointerGetter( j )->IsSlave() )
                {
                    BiharmonicCodimensionInterfaceVisitor<N, T> biharmonic_interface;
                    i->EdgePointerGetter( j )->Accept( biharmonic_interface );

                    int slave_id = biharmonic_interface.SlaveID();
                    int master_id = biharmonic_interface.MasterID();
                    int slave_starting_dof = 0;
                    int master_starting_dof = 0;

                    auto constraint_data = biharmonic_interface.ConstraintData();
                    auto vertices_constraint_data = biharmonic_interface.VerticesConstraintData();
                    auto vertices_distri_data = biharmonic_interface.VerticesDistributionConstraintData();

                    std::for_each( vertices_distri_data[0].first._colIndices->begin(),
                                   vertices_distri_data[0].first._colIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( vertices_distri_data[0].second._colIndices->begin(),
                                   vertices_distri_data[0].second._colIndices->end(),
                                   [&]( int& index ) { index += master_starting_dof; } );
                    std::for_each( vertices_distri_data[1].first._colIndices->begin(),
                                   vertices_distri_data[1].first._colIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( vertices_distri_data[1].second._colIndices->begin(),
                                   vertices_distri_data[1].second._colIndices->end(),
                                   [&]( int& index ) { index += master_starting_dof; } );

                    {
                        Eigen::SparseVector<T> vsv1( total_dof ), vsv2( total_dof ), vsv3( total_dof ), vsv4( total_dof );
                        for ( auto it = vertices_distri_data[0].first._colIndices->begin();
                              it != vertices_distri_data[0].first._colIndices->end(); ++it )
                        {
                            vsv1.coeffRef( *it ) = ( *vertices_distri_data[0].first._matrix )(
                                0, it - vertices_distri_data[0].first._colIndices->begin() );
                            vsv2.coeffRef( *it ) = ( *vertices_distri_data[0].first._matrix )(
                                1, it - vertices_distri_data[0].first._colIndices->begin() );
                        }
                        for ( auto it = vertices_distri_data[0].second._colIndices->begin();
                              it != vertices_distri_data[0].second._colIndices->end(); ++it )
                        {
                            vsv1.coeffRef( *it ) = -( *vertices_distri_data[0].second._matrix )(
                                0, it - vertices_distri_data[0].second._colIndices->begin() );
                            vsv2.coeffRef( *it ) = -( *vertices_distri_data[0].second._matrix )(
                                1, it - vertices_distri_data[0].second._colIndices->begin() );
                        }
                        for ( auto it = vertices_distri_data[1].first._colIndices->begin();
                              it != vertices_distri_data[1].first._colIndices->end(); ++it )
                        {
                            vsv3.coeffRef( *it ) = ( *vertices_distri_data[1].first._matrix )(
                                0, it - vertices_distri_data[1].first._colIndices->begin() );
                            vsv4.coeffRef( *it ) = ( *vertices_distri_data[1].first._matrix )(
                                1, it - vertices_distri_data[1].first._colIndices->begin() );
                        }
                        for ( auto it = vertices_distri_data[1].second._colIndices->begin();
                              it != vertices_distri_data[1].second._colIndices->end(); ++it )
                        {
                            vsv3.coeffRef( *it ) = -( *vertices_distri_data[1].second._matrix )(
                                0, it - vertices_distri_data[1].second._colIndices->begin() );
                            vsv4.coeffRef( *it ) = -( *vertices_distri_data[1].second._matrix )(
                                1, it - vertices_distri_data[1].second._colIndices->begin() );
                        }
                        _vertices_constraints.push_back( vsv1 );
                        _vertices_constraints.push_back( vsv2 );
                        _vertices_constraints.push_back( vsv3 );
                        _vertices_constraints.push_back( vsv4 );
                    }

                    std::for_each( constraint_data._rowIndices->begin(), constraint_data._rowIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( constraint_data._colIndices->begin(), constraint_data._colIndices->end(),
                                   [&]( int& index ) { index += master_starting_dof; } );
                    std::for_each( vertices_constraint_data._rowIndices->begin(), vertices_constraint_data._rowIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( vertices_constraint_data._colIndices->begin(), vertices_constraint_data._colIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    _matrix_data_container.push_back( std::move( constraint_data ) );
                    _matrix_data_container.push_back( std::move( vertices_constraint_data ) );
                }
            }
        }
        for ( const auto& i : _matrix_data_container )
        {
            _involved_indices.insert( _involved_indices.end(), i._rowIndices->begin(), i._rowIndices->end() );
            _involved_indices.insert( _involved_indices.end(), i._colIndices->begin(), i._colIndices->end() );
        }
        std::sort( _involved_indices.begin(), _involved_indices.end() );
        _involved_indices.erase( unique( _involved_indices.begin(), _involved_indices.end() ), _involved_indices.end() );
    }

    void AssembleConstraint( Eigen::SparseMatrix<T, Eigen::RowMajor>& sparse_constraint_matrix )
    {
        std::vector<Eigen::SparseVector<T>> constraint_container;
        int total_dof = _dof.TotalDof();
        MatrixData<T> global_constraint;
        for ( auto& i : _matrix_data_container )
            // {
            //     global_constraint += i;
            // }

            // for ( auto m = global_constraint._rowIndices->begin(); m != global_constraint._rowIndices->end(); m++ )
            // {
            //     {
            //         Eigen::SparseVector<T> sparse_vector( total_dof );
            //         sparse_vector.coeffRef( *m ) = 1.0;
            //         for ( int n = 0; n < global_constraint._colIndices->size(); n++ )
            //         {
            //             sparse_vector.coeffRef( ( *global_constraint._colIndices )[n] ) =
            //                 -( *global_constraint._matrix )( m - global_constraint._rowIndices->begin(), n );
            //         }
            //         constraint_container.push_back( sparse_vector );
            //     }
            // }

            for ( auto& i : _matrix_data_container )
            {
                for ( auto m = i._rowIndices->begin(); m != i._rowIndices->end(); m++ )
                {
                    // if (std::find(_additional_constraint.begin(),
                    // _additional_constraint.end(), *m) == _additional_constraint.end())
                    {
                        Eigen::SparseVector<T> sparse_vector( total_dof );
                        sparse_vector.coeffRef( *m ) = 1.0;
                        for ( int n = 0; n < i._colIndices->size(); n++ )
                        {
                            // if (std::find(_additional_constraint.begin(),
                            // _additional_constraint.end(), (*i._colIndices)[n]) ==
                            // _additional_constraint.end())
                            {
                                sparse_vector.coeffRef( ( *i._colIndices )[n] ) =
                                    -( *i._matrix )( m - i._rowIndices->begin(), n );
                            }
                        }
                        constraint_container.push_back( sparse_vector );
                    }
                }
            }
        sparse_constraint_matrix.resize( constraint_container.size(), _dof.TotalDof() );
        for ( int i = 0; i < constraint_container.size(); i++ )
        {
            sparse_constraint_matrix.row( i ) = constraint_container[i].transpose();
        }
    }

    void AssembleConstraintWithAdditionalConstraint( Eigen::SparseMatrix<T, Eigen::RowMajor>& sparse_constraint_matrix )
    {
        AssembleConstraint( sparse_constraint_matrix );
        int original_constraint_size = sparse_constraint_matrix.rows();
        sparse_constraint_matrix.conservativeResize( sparse_constraint_matrix.rows() + _additional_constraint.size(),
                                                     sparse_constraint_matrix.cols() );
        for ( int i = 0; i < _additional_constraint.size(); i++ )
        {
            sparse_constraint_matrix.coeffRef( original_constraint_size, _additional_constraint[i] ) = 1;
            original_constraint_size++;
        }
    }

    void AssembleByReducedKernel( Eigen::SparseMatrix<T>& sparse_kernel_matrix )
    {
        std::vector<Eigen::SparseVector<T>> constraint_container;
        std::vector<Eigen::SparseVector<T>> kernel_vector_container;
        int total_dof = _dof.TotalDof();

        for ( auto& i : _matrix_data_container )
        {
            for ( auto m = i._rowIndices->begin(); m != i._rowIndices->end(); )
            {
                auto it = std::find( _vertex_indices.begin(), _vertex_indices.end(), *m );
                if ( it != _vertex_indices.end() )
                {
                    if ( std::find( _additional_constraint.begin(), _additional_constraint.end(), *m ) ==
                         _additional_constraint.end() )
                    {
                        Eigen::SparseVector<T> sparse_vector( total_dof );
                        sparse_vector.coeffRef( *m ) = 1.0;
                        for ( int n = 0; n < i._colIndices->size(); n++ )
                        {
                            sparse_vector.coeffRef( ( *i._colIndices )[n] ) = -( *i._matrix )( m - i._rowIndices->begin(), n );
                        }
                        constraint_container.push_back( sparse_vector );
                    }
                    else
                    {
                        _vertex_indices.erase( it );
                    }
                    i.RowRemove( m - i._rowIndices->begin() );
                }
                else
                {
                    m++;
                }
            }
        }

        MatrixData<T> global;
        for ( auto& i : _matrix_data_container )
        {
            global = global + i;
        }
        std::vector<int> active_boundary_constraint;
        std::set_intersection( _additional_constraint.begin(), _additional_constraint.end(), _involved_indices.begin(),
                               _involved_indices.end(), std::back_inserter( active_boundary_constraint ) );

        for ( auto n = global._colIndices->begin(); n != global._colIndices->end(); n++ )
        {
            auto it = std::find( active_boundary_constraint.begin(), active_boundary_constraint.end(), *n );
            if ( it == active_boundary_constraint.end() )
            {
                Eigen::SparseVector<T> sparse_vector( total_dof );
                sparse_vector.coeffRef( *n ) = 1.0;
                for ( int m = 0; m < global._rowIndices->size(); m++ )
                {
                    sparse_vector.coeffRef( ( *global._rowIndices )[m] ) =
                        ( *global._matrix )( m, n - global._colIndices->begin() );
                }
                kernel_vector_container.push_back( sparse_vector );
            }
            else
            {
                active_boundary_constraint.erase( it );
            }
        }

        std::vector<int> pure_slave_vertex_indices;

        std::set_difference( _vertex_indices.begin(), _vertex_indices.end(), global._colIndices->begin(),
                             global._colIndices->end(), std::back_inserter( pure_slave_vertex_indices ) );

        for ( const auto& i : pure_slave_vertex_indices )
        {
            Eigen::SparseVector<T> sparse_vector( total_dof );
            sparse_vector.coeffRef( i ) = 1.0;
            kernel_vector_container.push_back( sparse_vector );
        }
        if ( constraint_container.size() == 0 )
        {
            for ( auto it = kernel_vector_container.begin(); it != kernel_vector_container.end(); it++ )
            {
                sparse_kernel_matrix.conservativeResize( _dof.TotalDof(), sparse_kernel_matrix.cols() + 1 );
                sparse_kernel_matrix.rightCols( 1 ) = *it;
            }
        }
        else
        {
            Eigen::SparseMatrix<T, Eigen::RowMajor> sparse_constraint_matrix;
            sparse_constraint_matrix.resize( constraint_container.size(), _dof.TotalDof() );
            for ( int i = 0; i < constraint_container.size(); i++ )
            {
                sparse_constraint_matrix.row( i ) = constraint_container[i].transpose();
            }

            auto pre_kernel_it = std::partition( kernel_vector_container.begin(), kernel_vector_container.end(),
                                                 [&sparse_constraint_matrix]( const Eigen::SparseVector<T>& ii ) {
                                                     return ( sparse_constraint_matrix * ii ).norm() > 0;
                                                 } );
            Eigen::SparseMatrix<T, Eigen::ColMajor> sparse_pre_kernel_matrix;
            sparse_pre_kernel_matrix.resize( _dof.TotalDof(), pre_kernel_it - kernel_vector_container.begin() );
            for ( int i = 0; i < pre_kernel_it - kernel_vector_container.begin(); i++ )
            {
                sparse_pre_kernel_matrix.col( i ) = kernel_vector_container[i];
            }
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dense_constraint_matrix = sparse_constraint_matrix * sparse_pre_kernel_matrix;

            // LU kernel
            Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> lu_decomp( dense_constraint_matrix );
            lu_decomp.setThreshold( 1e-10 );
            sparse_kernel_matrix = ( sparse_pre_kernel_matrix * lu_decomp.kernel() ).sparseView();

            // SVD kernel
            // Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd( dense_constraint_matrix, Eigen::ComputeThinU | Eigen::ComputeFullV );
            // // std::cout << svd.singularValues().transpose() << std::endl;
            // int count{0};
            // for ( int i = 0; i < svd.singularValues().size(); ++i )
            // {
            //     if ( svd.singularValues()( i ) < 1e-12 )
            //     {
            //         break;
            //     }
            //     count++;
            // }
            // sparse_kernel_matrix = ( sparse_pre_kernel_matrix * svd.matrixV().rightCols( svd.matrixV().cols() - count ) ).sparseView( 1e-15 );

            for ( auto it = pre_kernel_it; it != kernel_vector_container.end(); it++ )
            {
                sparse_kernel_matrix.conservativeResize( _dof.TotalDof(), sparse_kernel_matrix.cols() + 1 );
                sparse_kernel_matrix.rightCols( 1 ) = *it;
            }
        }

        std::vector<int> dof_indices( total_dof );
        std::iota( dof_indices.begin(), dof_indices.end(), 0 );
        std::vector<int> idle_index;
        std::set_difference( dof_indices.begin(), dof_indices.end(), _involved_indices.begin(), _involved_indices.end(),
                             std::back_inserter( idle_index ) );

        std::vector<int> idle_index_constrained;
        std::set_difference( idle_index.begin(), idle_index.end(), _additional_constraint.begin(),
                             _additional_constraint.end(), std::back_inserter( idle_index_constrained ) );

        for ( const auto& i : idle_index_constrained )
        {
            sparse_kernel_matrix.conservativeResize( _dof.TotalDof(), sparse_kernel_matrix.cols() + 1 );
            Eigen::SparseVector<T> sparse_vector( total_dof );
            sparse_vector.coeffRef( i ) = 1.0;
            sparse_kernel_matrix.rightCols( 1 ) = sparse_vector;
        }
    }

    void Additional_Constraint( const std::vector<int>& indices )
    {
        _additional_constraint = indices;
    }

    void AssembleByCodimension( Eigen::SparseMatrix<T>& sparse_kernel_matrix )
    {
        std::vector<Eigen::SparseVector<T>> basis_container;
        int total_dof = _dof.TotalDof();

        MatrixData<T> global_constraint;
        for ( auto& i : _matrix_data_container )
        {
            global_constraint += i;
        }
        for ( auto n = global_constraint._colIndices->begin(); n != global_constraint._colIndices->end(); n++ )
        {
            if ( std::find( _additional_constraint.begin(), _additional_constraint.end(), *n ) == _additional_constraint.end() )
            {
                Eigen::SparseVector<T> sparse_vector( total_dof );
                sparse_vector.coeffRef( *n ) = 1.0;
                for ( int m = 0; m < global_constraint._rowIndices->size(); m++ )
                {
                    sparse_vector.coeffRef( ( *global_constraint._rowIndices )[m] ) =
                        ( *global_constraint._matrix )( m, n - global_constraint._colIndices->begin() );
                }
                basis_container.push_back( std::move( sparse_vector ) );
            }
        }

        Eigen::SparseMatrix<T, Eigen::RowMajor> sparse_vertices_constraints( _vertices_constraints.size(), total_dof );
        for ( int i = 0; i < _vertices_constraints.size(); i++ )
        {
            sparse_vertices_constraints.row( i ) = _vertices_constraints[i].transpose();
        }
        sparse_vertices_constraints.prune( 1e-12 );

        auto pre_kernel_it = std::partition( basis_container.begin(), basis_container.end(),
                                             [&sparse_vertices_constraints]( const Eigen::SparseVector<T>& ii ) {
                                                 return ( sparse_vertices_constraints * ii ).norm() > 0;
                                             } );
        Eigen::SparseMatrix<T, Eigen::ColMajor> sparse_pre_kernel_matrix;
        sparse_pre_kernel_matrix.resize( _dof.TotalDof(), pre_kernel_it - basis_container.begin() );

        for ( int i = 0; i < pre_kernel_it - basis_container.begin(); i++ )
        {
            sparse_pre_kernel_matrix.col( i ) = basis_container[i];
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dense_constraint_matrix = sparse_vertices_constraints * sparse_pre_kernel_matrix;

        // LU kernel
        Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> lu_decomp( dense_constraint_matrix );
        lu_decomp.setThreshold( 3e-13 );
        sparse_kernel_matrix = ( sparse_pre_kernel_matrix * lu_decomp.kernel() ).sparseView();

        for ( auto it = pre_kernel_it; it != basis_container.end(); it++ )
        {
            sparse_kernel_matrix.conservativeResize( _dof.TotalDof(), sparse_kernel_matrix.cols() + 1 );
            sparse_kernel_matrix.rightCols( 1 ) = *it;
        }

        std::vector<int> dof_indices( total_dof );
        std::iota( dof_indices.begin(), dof_indices.end(), 0 );

        std::vector<int> idle_index;
        std::set_difference( dof_indices.begin(), dof_indices.end(), _involved_indices.begin(), _involved_indices.end(),
                             std::back_inserter( idle_index ) );

        std::vector<int> idle_index_constrained;
        std::set_difference( idle_index.begin(), idle_index.end(), _additional_constraint.begin(),
                             _additional_constraint.end(), std::back_inserter( idle_index_constrained ) );

        for ( const auto& i : idle_index_constrained )
        {
            sparse_kernel_matrix.conservativeResize( _dof.TotalDof(), sparse_kernel_matrix.cols() + 1 );
            Eigen::SparseVector<T> sparse_vector( total_dof );
            sparse_vector.coeffRef( i ) = 1.0;
            sparse_kernel_matrix.rightCols( 1 ) = sparse_vector;
        }
    }

protected:
    const DofMapper _dof;
    std::vector<MatrixData<T>> _matrix_data_container;
    std::vector<int> _vertex_indices;
    std::vector<int> _involved_indices;
    std::vector<int> _additional_constraint;
    std::vector<Eigen::SparseVector<T>> _vertices_constraints;
};

template <typename T>
class KLShellConstraintAssembler
{
public:
    KLShellConstraintAssembler( DofMapper& dof ) : _dof( dof )
    {
    }

    void ConstraintInitialize( const std::vector<std::shared_ptr<Surface<3, T>>>& cells )
    {
        for ( auto& i : cells )
        {
            for ( int j = 0; j < 4; j++ )
            {
                if ( i->EdgePointerGetter( j )->IsMatched() && i->EdgePointerGetter( j )->IsSlave() )
                {
                    _visitor_map[std::make_pair( i->EdgePointerGetter( j )->GetID(),
                                                 i->EdgePointerGetter( j )->Counterpart().lock()->GetID() )];
                }
            }
        }
    }

    void ConstraintCreator( const std::vector<std::shared_ptr<Surface<3, T>>>& cells )
    {
        _matrix_data_container.clear();
        _additional_constraint.clear();
        for ( auto& i : cells )
        {
            for ( int j = 0; j < 4; j++ )
            {
                if ( i->EdgePointerGetter( j )->IsMatched() && i->EdgePointerGetter( j )->IsSlave() )
                {
                    auto it = _visitor_map.find( std::make_pair(
                        i->EdgePointerGetter( j )->GetID(), i->EdgePointerGetter( j )->Counterpart().lock()->GetID() ) );
                    ASSERT( it != _visitor_map.end(), " constraint visitor not initialized.\n" );

                    i->EdgePointerGetter( j )->Accept( it->second );
                    int slave_id = it->second.SlaveID();
                    int master_id = it->second.MasterID();
                    int slave_starting_dof = _dof.StartingDof( slave_id );
                    int master_starting_dof = _dof.StartingDof( master_id );

                    auto constraint_data = it->second.ConstraintData();
                    auto vertices_constraint_data = it->second.VerticesConstraintData();
                    std::for_each( constraint_data._rowIndices->begin(), constraint_data._rowIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( constraint_data._colIndices->begin(), constraint_data._colIndices->end(),
                                   [&]( int& index ) { index += master_starting_dof; } );
                    std::for_each( vertices_constraint_data._rowIndices->begin(), vertices_constraint_data._rowIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    std::for_each( vertices_constraint_data._colIndices->begin(), vertices_constraint_data._colIndices->end(),
                                   [&]( int& index ) { index += slave_starting_dof; } );
                    _matrix_data_container.push_back( std::move( constraint_data ) );
                    _matrix_data_container.push_back( std::move( vertices_constraint_data ) );
                }
            }
        }
    }
    void Additional_Constraint( const std::vector<int>& indices )
    {
        _additional_constraint = indices;
    }

    void AssembleConstraints( Eigen::SparseMatrix<T>& sparse_kernel_matrix )
    {
        std::vector<Eigen::SparseVector<T>> basis_container;
        int total_dof = _dof.TotalDof();

        MatrixData<T> global_constraint;
        for ( auto& i : _matrix_data_container )
        {
            global_constraint += i;
        }

        std::vector<Eigen::Triplet<T>> constraint_triplets;
        int basis_size = 0;
        for ( int i = 0; i < total_dof; ++i )
        {
            if ( std::find( _additional_constraint.begin(), _additional_constraint.end(), i ) != _additional_constraint.end() ||
                 std::find( global_constraint._rowIndices->begin(), global_constraint._rowIndices->end(), i ) !=
                     global_constraint._rowIndices->end() )
            {
                continue;
            }
            else
            {
                constraint_triplets.push_back( Eigen::Triplet<T>( i, basis_size, 1 ) );
                int pos = std::find( global_constraint._colIndices->begin(), global_constraint._colIndices->end(), i ) -
                          global_constraint._colIndices->begin();
                if ( pos < global_constraint._colIndices->size() )
                {
                    for ( int j = 0; j < global_constraint._rowIndices->size(); ++j )
                    {
                        constraint_triplets.push_back( Eigen::Triplet<T>( ( *global_constraint._rowIndices )[j], basis_size,
                                                                          -( *global_constraint._matrix )( j, pos ) ) );
                    }
                }
                basis_size++;
            }
        }
        sparse_kernel_matrix.resize( total_dof, basis_size );
        sparse_kernel_matrix.setFromTriplets( constraint_triplets.begin(), constraint_triplets.end() );
    }

protected:
    const DofMapper _dof;
    std::vector<MatrixData<T>> _matrix_data_container;
    std::vector<int> _additional_constraint;
    std::map<std::pair<int, int>, KLShellC1InterfaceVisitor<T>> _visitor_map;
};