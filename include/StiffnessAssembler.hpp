#pragma once
#include "StiffnessVisitor.hpp"
#include "Surface.hpp"

template <typename T>
class StiffnessAssembler
{
public:
    using LoadFunctor = typename T::LoadFunctor;
    using DataType = typename T::DataType;
    using Knot = typename T::Knot;
    static const int Dim = T::Dim;

    StiffnessAssembler( DofMapper& dof ) : _dof( dof )
    {
    }

    template <typename T1>
    void Assemble( const T1& cells,
                   const LoadFunctor& load,
                   Eigen::SparseMatrix<DataType>& stiffness_matrix,
                   Eigen::SparseMatrix<DataType>& load_vector )
    {
        std::vector<std::unique_ptr<T>> bihamronic_visitors;
        for ( auto& i : cells )
        {
            std::unique_ptr<T> biharmonic( new T( load ) );
            if ( _historyData )
            {
                biharmonic->SetStateDatas( _c + _dof.StartingDof( i->GetID() ) * Dim, _ct + _dof.StartingDof( i->GetID() ) * Dim );
            }
            biharmonic->ThreadSetter( _thd );
            i->Accept( *biharmonic );
            bihamronic_visitors.push_back( std::move( biharmonic ) );
        }
        std::vector<Eigen::Triplet<DataType>> stiffness_triplet, load_triplet;

        for ( auto& i : bihamronic_visitors )
        {
            const auto stiffness_triplet_in = i->GetStiffness();
            const auto load_triplet_in = i->GetRhs();
            int id = i->ID();
            int starting_dof = _dof.StartingDof( id );
            for ( const auto& j : stiffness_triplet_in )
            {
                stiffness_triplet.push_back( Eigen::Triplet<DataType>( starting_dof + j.row(), starting_dof + j.col(), j.value() ) );
            }
            for ( const auto& j : load_triplet_in )
            {
                load_triplet.push_back( Eigen::Triplet<DataType>( starting_dof + j.row(), j.col(), j.value() ) );
            }
        }
        stiffness_matrix.setFromTriplets( stiffness_triplet.begin(), stiffness_triplet.end() );
        load_vector.setFromTriplets( load_triplet.begin(), load_triplet.end() );
    }

    template <typename T1>
    void Assemble( const T1& cells, Eigen::SparseMatrix<DataType>& stiffness_matrix )
    {
        std::vector<std::unique_ptr<T>> bihamronic_visitors;
        const LoadFunctor load = []( const Knot& u ) { return std::vector<DataType>{0, 0, 0}; };
        for ( auto& i : cells )
        {
            std::unique_ptr<T> biharmonic( new T( load ) );
            if ( _historyData )
            {
                biharmonic->SetStateDatas( _c + _dof.StartingDof( i->GetID() ) * Dim, _ct + _dof.StartingDof( i->GetID() ) * Dim );
            }
            biharmonic->ThreadSetter( _thd );
            i->Accept( *biharmonic );
            bihamronic_visitors.push_back( std::move( biharmonic ) );
        }
        std::vector<Eigen::Triplet<DataType>> stiffness_triplet, load_triplet;

        for ( auto& i : bihamronic_visitors )
        {
            const auto stiffness_triplet_in = i->GetStiffness();
            const auto load_triplet_in = i->GetRhs();
            int id = i->ID();
            int starting_dof = _dof.StartingDof( id );
            for ( const auto& j : stiffness_triplet_in )
            {
                stiffness_triplet.push_back( Eigen::Triplet<DataType>( starting_dof + j.row(), starting_dof + j.col(), j.value() ) );
            }
            for ( const auto& j : load_triplet_in )
            {
                load_triplet.push_back( Eigen::Triplet<DataType>( starting_dof + j.row(), j.col(), j.value() ) );
            }
        }
        // Eigen::SparseMatrix<DataType> triangle_stiffness_matrix;
        // triangle_stiffness_matrix.resize( Dim * _dof.TotalDof(), Dim * _dof.TotalDof() );
        // triangle_stiffness_matrix.setFromTriplets( stiffness_triplet.begin(), stiffness_triplet.end() );
        // stiffness_matrix += triangle_stiffness_matrix.template selfadjointView<Eigen::Upper>();

        stiffness_matrix.setFromTriplets( stiffness_triplet.begin(), stiffness_triplet.end() );
    }

    void SetStateDatas( DataType* disp, DataType* vel )
    {
        _c = disp;
        _ct = vel;
        _historyData = true;
    }
    void ThreadSetter( int thd )
    {
        _thd = thd;
    }

protected:
    const DofMapper _dof;
    DataType* _c;
    DataType* _ct;
    bool _historyData{false};
    int _thd = 8;
};