#include "BiharmonicDirichletBoundaryVisitor.hpp"
#include "DofMapper.hpp"
#include "Surface.hpp"

template <typename T>
class BoundaryAssembler
{
public:
    static constexpr int PhyDim = T::PhyDim;
    using DataType = typename T::DataType;
    using LoadFunctor = typename BiharmonicDirichletBoundaryVisitor<PhyDim, DataType>::LoadFunctor;
    BoundaryAssembler( DofMapper& dof ) : _dof( dof )
    {
    }
    void BoundaryValueCreator( const std::vector<std::shared_ptr<T>>& cells, const LoadFunctor& load_func, Eigen::SparseMatrix<DataType>& load_vector )
    {
        _boundaryValues.clear();
        for ( auto& i : cells )
        {
            BiharmonicDirichletBoundaryVisitor<PhyDim, DataType> biharmonic_boundary( load_func );
            for ( int j = 0; j < 4; j++ )
            {
                if ( !i->EdgePointerGetter( j )->IsMatched() )
                {
                    i->EdgePointerGetter( j )->Accept( biharmonic_boundary );
                }
            }
            if ( biharmonic_boundary.SolveDirichletBoundary() )
            {
                auto res = biharmonic_boundary.DirichletBoundaryValue();
                int starting_dof = _dof.StartingDof( i->GetID() );
                for ( const auto& i : res )
                {
                    _boundaryValues.push_back( std::make_pair( i.first + starting_dof, i.second ) );
                }
            }
        }
        load_vector.resize( _dof.TotalDof(), 1 );
        for ( auto& i : _boundaryValues )
        {
            load_vector.insert( i.first, 0 ) = i.second;
        }
    }

protected:
    const DofMapper _dof;
    std::vector<std::pair<int, DataType>> _boundaryValues;
};