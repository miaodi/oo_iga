//
// Created by di miao on 10/24/17.
//

#pragma once

#include "InterfaceVisitor.hpp"
#include "PoissonInterfaceVisitor.hpp"
#include <unsupported/Eigen/KroneckerProduct>

template <int N, typename T>
class BiharmonicInterfaceVisitor : public InterfaceVisitor<N, T>
{
public:
    using Knot = typename InterfaceVisitor<N, T>::Knot;
    using Quadrature = typename InterfaceVisitor<N, T>::Quadrature;
    using QuadList = typename InterfaceVisitor<N, T>::QuadList;
    using KnotSpan = typename InterfaceVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename InterfaceVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename InterfaceVisitor<N, T>::LoadFunctor;
    using Matrix = typename InterfaceVisitor<N, T>::Matrix;
    using Vector = typename InterfaceVisitor<N, T>::Vector;
    using DomainShared_ptr = typename InterfaceVisitor<N, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename InterfaceVisitor<N, T>::ConstraintIntegralElementAssembler;

public:
    BiharmonicInterfaceVisitor( std::unique_ptr<PoissonInterfaceVisitor<N, T>> ptr = std::make_unique<PoissonInterfaceVisitor<N, T>>() )
        : InterfaceVisitor<N, T>()
    {
        _poisson = std::move( ptr );
    }

    void Visit( Element<1, N, T>* g );

protected:
    void SolveConstraint( Edge<N, T>* );

    void LocalAssemble( Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan& );

    template <int n = N>
    typename std::enable_if<n == 3, void>::type C1IntegralElementAssembler( Matrix& slave_constraint_basis,
                                                                            std::vector<int>& slave_constraint_basis_indices,
                                                                            Matrix& master_constrint_basis,
                                                                            std::vector<int>& master_constraint_basis_indices,
                                                                            Matrix& multiplier_basis,
                                                                            std::vector<int>& multiplier_basis_indices,
                                                                            T& integral_weight,
                                                                            Edge<N, T>* edge,
                                                                            const Quadrature& u );

    template <int n = N>
    typename std::enable_if<n == 2, void>::type C1IntegralElementAssembler( Matrix& slave_constraint_basis,
                                                                            std::vector<int>& slave_constraint_basis_indices,
                                                                            Matrix& master_constrint_basis,
                                                                            std::vector<int>& master_constraint_basis_indices,
                                                                            Matrix& multiplier_basis,
                                                                            std::vector<int>& multiplier_basis_indices,
                                                                            T& integral_weight,
                                                                            Edge<N, T>* edge,
                                                                            const Quadrature& u );

protected:
    std::vector<Eigen::Triplet<T>> _c1Slave;
    std::vector<Eigen::Triplet<T>> _c1Master;
    std::unique_ptr<PoissonInterfaceVisitor<N, T>> _poisson;
};

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::Visit( Element<1, N, T>* g )
{
    _poisson->Visit( g );
    InterfaceVisitor<N, T>::Visit( g );
}

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::SolveConstraint( Edge<N, T>* edge )
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> slave_indices = Accessory::ColIndicesVector( _c1Slave );
    std::vector<int> master_indices = Accessory::ColIndicesVector( _c1Master );
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector( _c1Slave );
    std::vector<int> c0_slave_indices = *( _poisson->ConstraintData()._rowIndices );

    std::vector<int> c1_slave_indices;
    std::set_difference( slave_indices.begin(), slave_indices.end(), c0_slave_indices.begin(), c0_slave_indices.end(),
                         std::back_inserter( c1_slave_indices ) );

    auto c1_slave_indices_inverse_map = Accessory::IndicesInverseMap( c1_slave_indices );
    auto c0_slave_indices_inverse_map = Accessory::IndicesInverseMap( c0_slave_indices );
    auto master_indices_inverse_map = Accessory::IndicesInverseMap( master_indices );
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap( multiplier_indices );

    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs, condensed_c0_slave;
    this->CondensedTripletVia( multiplier_indices_inverse_map, c1_slave_indices_inverse_map, _c1Slave, condensed_gramian );
    this->CondensedTripletVia( multiplier_indices_inverse_map, master_indices_inverse_map, _c1Master, condensed_rhs );
    this->CondensedTripletVia( multiplier_indices_inverse_map, c0_slave_indices_inverse_map, _c1Slave, condensed_c0_slave );

    Matrix gramian_matrix, rhs_matrix, c0_slave_matrix;
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), c1_slave_indices_inverse_map.size(), condensed_gramian, gramian_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), master_indices_inverse_map.size(), condensed_rhs, rhs_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), c0_slave_indices_inverse_map.size(),
                           condensed_c0_slave, c0_slave_matrix );
    // Accessory::removeNoise( gramian_matrix, 1e-7 * abs( gramian_matrix( 0, 0 ) ) );
    // Accessory::removeNoise( rhs_matrix, 1e-14 );
    // Accessory::removeNoise( c0_slave_matrix, 1e-7 * abs( c0_slave_matrix( 0, 0 ) ) );

    Matrix c1_constraint = this->SolveNonSymmetric( gramian_matrix, rhs_matrix );
    Matrix c0_c1_constraint = this->SolveNonSymmetric( gramian_matrix, -c0_slave_matrix );
    auto c1_slave_indices_copy = c1_slave_indices;
    MatrixData<T> c1_constraint_data( c1_constraint, c1_slave_indices, master_indices );
    MatrixData<T> c0_c1constraint_data( c0_c1_constraint, c1_slave_indices_copy, c0_slave_indices );
    auto c0_slave_c1_constraint_data = c0_c1constraint_data * _poisson->ConstraintData();
    this->_constraintData = c1_constraint_data + c0_slave_c1_constraint_data + _poisson->ConstraintData();
}

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::LocalAssemble( Element<1, N, T>* g, const QuadratureRule<T>& quadrature_rule, const KnotSpan& knot_span )
{
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto c1_function =
        std::bind( &BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler<>, this, _1, _2, _3, _4, _5, _6, _7, _8, _9 );
    this->ConstraintLocalAssemble( g, quadrature_rule, knot_span, c1_function, _c1Slave, _c1Master );
}

template <int N, typename T>
template <int n>
typename std::enable_if<n == 3, void>::type BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler(
    Matrix& slave_constraint_basis,
    std::vector<int>& slave_constraint_basis_indices,
    Matrix& master_constraint_basis,
    std::vector<int>& master_constraint_basis_indices,
    Matrix& multiplier_basis,
    std::vector<int>& multiplier_basis_indices,
    T& integral_weight,
    Edge<N, T>* edge,
    const Quadrature& u )
{
    auto multiplier_domain = edge->GetDomain();
    auto slave_domain = edge->Parent( 0 ).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent( 0 ).lock()->GetDomain();

    //    set up integration weights
    integral_weight = u.second;

    // Map abscissa from Lagrange multiplier space to slave and master domain
    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    // Evaluate derivative upto 1^st order in slave and master domain
    auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 1 );
    auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 1 );

    //  Evaluate Lagrange multiplier basis
    auto multiplier_evals = multiplier_domain->EvalDualAllTensor( u.first );

    // Resize integration matrices
    slave_constraint_basis.resize( 1, slave_evals->size() );
    master_constraint_basis.resize( 1, master_evals->size() );
    multiplier_basis.resize( 1, multiplier_evals->size() );

    // Compute the following matrix
    // +-----------+-----------+
    // | ∂ξ_m/∂ξ_s | ∂η_m/∂ξ_s |
    // +-----------+-----------+
    // | ∂ξ_m/∂η_s | ∂η_m/∂η_s |
    // +-----------+-----------+
    Matrix slave_jacobian = slave_domain->JacobianMatrix( slave_quadrature_abscissa );
    Matrix master_jacobian = master_domain->JacobianMatrix( master_quadrature_abscissa );
    // Matrix master_to_slave = slave_jacobian * master_jacobian.inverse();
    Eigen::Matrix<T, 3, 1> s_s, s_t, m_s, m_t, s_n, m_n;
    s_s = slave_jacobian.col( 0 );
    s_t = slave_jacobian.col( 1 );
    m_s = master_jacobian.col( 0 );
    m_t = master_jacobian.col( 1 );
    s_n = s_s.cross( s_t );
    m_n = m_s.cross( m_t );

    Eigen::Matrix<T, 3, 3> rotation_matrix = Accessory::RotationMatrix( m_n, s_n );

    Matrix gramian = master_jacobian.transpose() * master_jacobian;
    Matrix rhs = master_jacobian.transpose() * rotation_matrix.transpose() * slave_jacobian;

    Matrix sol = gramian.partialPivLu().solve( rhs );

    // Two strategies for horizontal edge and vertical edge.
    switch ( edge->GetOrient() )
    {
    // For south and north edge derivative w.r.t η_s should be consistent
    case Orientation::south:
    case Orientation::north:
    {
        for ( int j = 0; j < slave_evals->size(); ++j )
        {
            slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[2];
        }
        for ( int j = 0; j < master_evals->size(); ++j )
        {
            master_constraint_basis( 0, j ) =
                ( *master_evals )[j].second[1] * sol( 0, 1 ) + ( *master_evals )[j].second[2] * sol( 1, 1 );
        }
        break;
    }
    // For south and north edge derivative w.r.t ξ_s should be consistent
    case Orientation::east:
    case Orientation::west:
    {
        for ( int j = 0; j < slave_evals->size(); ++j )
        {
            slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[1];
        }
        for ( int j = 0; j < master_evals->size(); ++j )
        {
            master_constraint_basis( 0, j ) =
                ( *master_evals )[j].second[1] * sol( 0, 0 ) + ( *master_evals )[j].second[2] * sol( 1, 0 );
        }
        break;
    }
    }

    // Lagrange multiplier basis
    for ( int j = 0; j < multiplier_evals->size(); ++j )
    {
        multiplier_basis( 0, j ) = ( *multiplier_evals )[j].second[0];
    }

    Eigen::Matrix<T, 3, 3> identity;
    identity.setIdentity();

    master_constraint_basis = kroneckerProduct( master_constraint_basis, identity ).eval();
    slave_constraint_basis = kroneckerProduct( slave_constraint_basis, identity ).eval();
    multiplier_basis = kroneckerProduct( multiplier_basis, identity ).eval();

    master_constraint_basis = ( rotation_matrix * master_constraint_basis ).eval();

    // set up local indices corresponding to test basis functions and trial basis functions
    if ( slave_constraint_basis_indices.size() == 0 )
    {
        auto index = slave_domain->ActiveIndex( slave_quadrature_abscissa );
        for ( auto& i : index )
        {
            for ( int j = 0; j < 3; j++ )
            {
                slave_constraint_basis_indices.push_back( 3 * i + j );
            }
        }
    }
    if ( master_constraint_basis_indices.size() == 0 )
    {
        auto index = master_domain->ActiveIndex( master_quadrature_abscissa );
        for ( auto& i : index )
        {
            for ( int j = 0; j < 3; j++ )
            {
                master_constraint_basis_indices.push_back( 3 * i + j );
            }
        }
    }
    if ( multiplier_basis_indices.size() == 0 )
    {
        for ( auto& i : *multiplier_evals )
        {
            for ( int j = 0; j < 3; j++ )
            {
                multiplier_basis_indices.push_back( 3 * i.first + j );
            }
        }
    }
}

template <int N, typename T>
template <int n>
typename std::enable_if<n == 2, void>::type BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler(
    Matrix& slave_constraint_basis,
    std::vector<int>& slave_constraint_basis_indices,
    Matrix& master_constraint_basis,
    std::vector<int>& master_constraint_basis_indices,
    Matrix& multiplier_basis,
    std::vector<int>& multiplier_basis_indices,
    T& integral_weight,
    Edge<N, T>* edge,
    const Quadrature& u )
{
    auto multiplier_domain = edge->GetDomain();
    auto slave_domain = edge->Parent( 0 ).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent( 0 ).lock()->GetDomain();

    //    set up integration weights
    integral_weight = u.second;

    // Map abscissa from Lagrange multiplier space to slave and master domain
    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    // Evaluate derivative upto 1^st order in slave and master domain
    auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 1 );
    auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 1 );

    //  Evaluate Lagrange multiplier basis
    auto multiplier_evals = multiplier_domain->EvalDualAllTensor( u.first );
    // auto multiplier_evals = ( multiplier_domain->BasisGetter( 0 ) ).EvalCodimensionBezierDual( u.first( 0 ) );

    // auto knot_vector = ( multiplier_domain->BasisGetter( 0 ) ).Knots();
    // for ( auto it = knot_vector.begin(); it != knot_vector.end(); ++it )
    // {
    //     if ( *it != 0 )
    //     {
    //         knot_vector.erase( it );
    //         break;
    //     }
    // }
    // for ( auto it = knot_vector.begin(); it != knot_vector.end(); ++it )
    // {
    //     if ( *it != 0 )
    //     {
    //         knot_vector.erase( it );
    //         break;
    //     }
    // }
    // for ( auto it = knot_vector.end() - 1; it != knot_vector.begin(); --it )
    // {
    //     if ( *it != 1 )
    //     {
    //         knot_vector.erase( it );
    //         break;
    //     }
    // }
    // for ( auto it = knot_vector.end() - 1; it != knot_vector.begin(); --it )
    // {
    //     if ( *it != 1 )
    //     {
    //         knot_vector.erase( it );
    //         break;
    //     }
    // }
    // BsplineBasis<T> tmp( knot_vector );
    // auto multiplier_evals = tmp.EvalDerAll( u.first( 0 ), 0 );

    // Resize integration matrices
    slave_constraint_basis.resize( 1, slave_evals->size() );
    master_constraint_basis.resize( 1, master_evals->size() );
    multiplier_basis.resize( 1, multiplier_evals->size() );

    // Compute the following matrix
    // +-----------+-----------+
    // | ∂ξ_m/∂ξ_s | ∂η_m/∂ξ_s |
    // +-----------+-----------+
    // | ∂ξ_m/∂η_s | ∂η_m/∂η_s |
    // +-----------+-----------+
    // Why transpose?
    Matrix slave_jacobian = slave_domain->JacobianMatrix( slave_quadrature_abscissa ).transpose();
    Matrix master_jacobian = master_domain->JacobianMatrix( master_quadrature_abscissa ).transpose();

    // Substitute master coordinate of master basis by slave coordinate
    for ( auto& i : *master_evals )
    {
        Vector tmp =
            slave_jacobian * master_jacobian.partialPivLu().solve( ( Vector( 2 ) << i.second[1], i.second[2] ).finished() );
        i.second[1] = tmp( 0 );
        i.second[2] = tmp( 1 );
    }

    // Two strategies for horizontal edge and vertical edge.
    switch ( edge->GetOrient() )
    {
    // For south and north edge derivative w.r.t η_s should be consistent
    case Orientation::south:
    case Orientation::north:
    {
        for ( int j = 0; j < slave_evals->size(); ++j )
        {
            slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[2];
        }
        for ( int j = 0; j < master_evals->size(); ++j )
        {
            master_constraint_basis( 0, j ) = ( *master_evals )[j].second[2];
        }
        break;
    }
    // For south and north edge derivative w.r.t ξ_s should be consistent
    case Orientation::east:
    case Orientation::west:
    {
        for ( int j = 0; j < slave_evals->size(); ++j )
        {
            slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[1];
        }
        for ( int j = 0; j < master_evals->size(); ++j )
        {
            master_constraint_basis( 0, j ) = ( *master_evals )[j].second[1];
        }
        break;
    }
    }

    // Lagrange multiplier basis
    for ( int j = 0; j < multiplier_evals->size(); ++j )
    {
        multiplier_basis( 0, j ) = ( *multiplier_evals )[j].second[0];
    }

    // set up indices corresponding to test basis functions and trial basis functions
    if ( slave_constraint_basis_indices.size() == 0 )
    {
        slave_constraint_basis_indices = slave_domain->ActiveIndex( slave_quadrature_abscissa );
    }
    if ( master_constraint_basis_indices.size() == 0 )
    {
        master_constraint_basis_indices = master_domain->ActiveIndex( master_quadrature_abscissa );
    }
    if ( multiplier_basis_indices.size() == 0 )
    {
        for ( const auto& i : *multiplier_evals )
        {
            multiplier_basis_indices.push_back( i.first );
        }
    }
}

template <int N, typename T>
class BiharmonicCodimensionInterfaceVisitor : public BiharmonicInterfaceVisitor<N, T>
{
public:
    using Knot = typename BiharmonicInterfaceVisitor<N, T>::Knot;
    using Quadrature = typename BiharmonicInterfaceVisitor<N, T>::Quadrature;
    using QuadList = typename BiharmonicInterfaceVisitor<N, T>::QuadList;
    using KnotSpan = typename BiharmonicInterfaceVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename BiharmonicInterfaceVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename BiharmonicInterfaceVisitor<N, T>::LoadFunctor;
    using Matrix = typename BiharmonicInterfaceVisitor<N, T>::Matrix;
    using Vector = typename BiharmonicInterfaceVisitor<N, T>::Vector;
    using DomainShared_ptr = typename BiharmonicInterfaceVisitor<N, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename BiharmonicInterfaceVisitor<N, T>::ConstraintIntegralElementAssembler;

    using BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler;

public:
    BiharmonicCodimensionInterfaceVisitor( const int c = 2 )
        : BiharmonicInterfaceVisitor<N, T>( std::make_unique<PoissonCodimensionInterfaceVisitor<N, T>>( c ) ), _codimension{c}
    {
    }

    const MatrixData<T>& VerticesConstraintData() const
    {
        return _slaveMasterConstraintData;
    }
    const std::array<std::pair<MatrixData<T>, MatrixData<T>>, 2>& VerticesDistributionConstraintData() const
    {
        return _vertices_constraints;
    }

    void Visit( Element<1, N, T>* g )
    {
        BiharmonicInterfaceVisitor<N, T>::Visit( g );
        auto edge = dynamic_cast<Edge<N, T>*>( g );
        if ( !edge->IsMatched() || !edge->IsSlave() )
        {
            return;
        }

        // create vertices constraints
        // eval at starting/ending points
        auto multiplier_domain = edge->GetDomain();
        auto slave_domain = edge->Parent( 0 ).lock()->GetDomain();
        auto master_domain = edge->Counterpart().lock()->Parent( 0 ).lock()->GetDomain();

        Eigen::Matrix<T, Eigen::Dynamic, 1> u( 1 );

        for ( int point = 0; point <= 1; point++ )
        {
            u << static_cast<T>( point );
            // Map abscissa from Lagrange multiplier space to slave and master domain
            Vector slave_quadrature_abscissa, master_quadrature_abscissa;
            if ( !Accessory::MapParametricPoint( &*multiplier_domain, u, &*slave_domain, slave_quadrature_abscissa ) )
            {
                std::cout << "MapParametericPoint failed" << std::endl;
            }
            if ( !Accessory::MapParametricPoint( &*multiplier_domain, u, &*master_domain, master_quadrature_abscissa ) )
            {
                std::cout << "MapParametericPoint failed" << std::endl;
            }

            // Evaluate derivative upto 1^st order in slave and master domain
            auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 1 );
            auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 1 );

            // Resize integration matrices
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> slave_constraint_basis( 2, slave_evals->size() );
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> master_constraint_basis( 2, master_evals->size() );

            // Compute the following matrix
            // +-----------+-----------+
            // | ∂ξ_m/∂ξ_s | ∂η_m/∂ξ_s |
            // +-----------+-----------+
            // | ∂ξ_m/∂η_s | ∂η_m/∂η_s |
            // +-----------+-----------+
            // Why transpose?
            Matrix slave_jacobian = slave_domain->JacobianMatrix( slave_quadrature_abscissa ).transpose();
            Matrix master_jacobian = master_domain->JacobianMatrix( master_quadrature_abscissa ).transpose();

            // Substitute master coordinate of master basis by slave coordinate
            for ( auto& i : *master_evals )
            {
                Vector tmp = slave_jacobian *
                             master_jacobian.partialPivLu().solve( ( Vector( 2 ) << i.second[1], i.second[2] ).finished() );
                i.second[1] = tmp( 0 );
                i.second[2] = tmp( 1 );
            }

            // Two strategies for horizontal edge and vertical edge.
            switch ( edge->GetOrient() )
            {
            // For south and north edge derivative w.r.t η_s should be consistent
            case Orientation::south:
            case Orientation::north:
            {
                for ( int j = 0; j < slave_evals->size(); ++j )
                {
                    slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[0];
                    slave_constraint_basis( 1, j ) = ( *slave_evals )[j].second[2];
                }
                for ( int j = 0; j < master_evals->size(); ++j )
                {
                    master_constraint_basis( 0, j ) = ( *master_evals )[j].second[0];
                    master_constraint_basis( 1, j ) = ( *master_evals )[j].second[2];
                }
                break;
            }
            // For south and north edge derivative w.r.t ξ_s should be consistent
            case Orientation::east:
            case Orientation::west:
            {
                for ( int j = 0; j < slave_evals->size(); ++j )
                {
                    slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[0];
                    slave_constraint_basis( 1, j ) = ( *slave_evals )[j].second[1];
                }
                for ( int j = 0; j < master_evals->size(); ++j )
                {
                    master_constraint_basis( 0, j ) = ( *master_evals )[j].second[0];
                    master_constraint_basis( 1, j ) = ( *master_evals )[j].second[1];
                }
                break;
            }
            }
            std::vector<int> slave_constraint_basis_indices, master_constraint_basis_indices, lm_indices;
            for ( int j = 0; j < slave_evals->size(); ++j )
            {
                slave_constraint_basis_indices.push_back( ( *slave_evals )[j].first );
            }
            for ( int j = 0; j < master_evals->size(); ++j )
            {
                master_constraint_basis_indices.push_back( ( *master_evals )[j].first );
            }
            lm_indices = {0, 1};
            *( _vertices_constraints[point].first._rowIndices ) = lm_indices;
            *( _vertices_constraints[point].first._colIndices ) = slave_constraint_basis_indices;
            *( _vertices_constraints[point].first._matrix ) = slave_constraint_basis;
            *( _vertices_constraints[point].second._rowIndices ) = lm_indices;
            *( _vertices_constraints[point].second._colIndices ) = master_constraint_basis_indices;
            *( _vertices_constraints[point].second._matrix ) = master_constraint_basis;
        }
    }

protected:
    void SolveConstraint( Edge<N, T>* );

protected:
    MatrixData<T> _slaveMasterConstraintData;
    std::array<std::pair<MatrixData<T>, MatrixData<T>>, 2> _vertices_constraints;
    int _codimension;
};

template <int N, typename T>
void BiharmonicCodimensionInterfaceVisitor<N, T>::SolveConstraint( Edge<N, T>* edge )
{
    const int dimension = 1;
    auto vertices_indices = edge->VerticesIndices( dimension, _codimension - 1 );

    // iterate across the constraint equation container and obtain activated local indices and lagrange multiplier indices
    std::vector<int> slave_indices = Accessory::ColIndicesVector( this->_c1Slave );
    std::vector<int> master_indices = Accessory::ColIndicesVector( this->_c1Master );
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector( this->_c1Slave );

    auto poisson_ptr = dynamic_cast<PoissonCodimensionInterfaceVisitor<N, T>*>( this->_poisson.get() );
    std::vector<int> c0_slave_indices = *( poisson_ptr->ConstraintData()._rowIndices );

    std::vector<int> c1_slave_complement_indices;
    std::set_union( vertices_indices.begin(), vertices_indices.end(), c0_slave_indices.begin(), c0_slave_indices.end(),
                    std::back_inserter( c1_slave_complement_indices ) );
    std::vector<int> c1_slave_indices;
    std::set_difference( slave_indices.begin(), slave_indices.end(), c1_slave_complement_indices.begin(),
                         c1_slave_complement_indices.end(), std::back_inserter( c1_slave_indices ) );

    auto c1_slave_indices_inverse_map = Accessory::IndicesInverseMap( c1_slave_indices );
    auto c0_slave_indices_inverse_map = Accessory::IndicesInverseMap( c0_slave_indices );
    auto master_indices_inverse_map = Accessory::IndicesInverseMap( master_indices );
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap( multiplier_indices );
    auto vertices_indices_inverse_map = Accessory::IndicesInverseMap( vertices_indices );

    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs, condensed_c0_slave, condensed_vertices_rhs;
    this->CondensedTripletVia( multiplier_indices_inverse_map, c1_slave_indices_inverse_map, this->_c1Slave, condensed_gramian );
    this->CondensedTripletVia( multiplier_indices_inverse_map, master_indices_inverse_map, this->_c1Master, condensed_rhs );
    this->CondensedTripletVia( multiplier_indices_inverse_map, c0_slave_indices_inverse_map, this->_c1Slave, condensed_c0_slave );
    this->CondensedTripletVia( multiplier_indices_inverse_map, vertices_indices_inverse_map, this->_c1Slave, condensed_vertices_rhs );

    Matrix gramian_matrix, rhs_matrix, c0_slave_matrix, vertices_rhs_matrix;
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), c1_slave_indices_inverse_map.size(), condensed_gramian, gramian_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), master_indices_inverse_map.size(), condensed_rhs, rhs_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), c0_slave_indices_inverse_map.size(),
                           condensed_c0_slave, c0_slave_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), vertices_indices_inverse_map.size(),
                           condensed_vertices_rhs, vertices_rhs_matrix );

    // Accessory::removeNoise( gramian_matrix, 1e-7 * abs( vertices_rhs_matrix( 0, 0 ) ) );
    // Accessory::removeNoise( rhs_matrix, 1e-14 );
    // Accessory::removeNoise( c0_slave_matrix, 1e-7 * abs( c0_slave_matrix( 0, 0 ) ) );
    // Accessory::removeNoise( vertices_rhs_matrix, 1e-7 * abs( vertices_rhs_matrix( 0, 0 ) ) );

    Matrix c1_constraint = this->SolveNonSymmetric( gramian_matrix, rhs_matrix );
    Matrix c0_c1_constraint = this->SolveNonSymmetric( gramian_matrix, -c0_slave_matrix );
    Matrix vertices_constraint = this->SolveNonSymmetric( gramian_matrix, -vertices_rhs_matrix );
    auto c1_slave_indices_copy = c1_slave_indices;
    auto c1_slave_indices_copy_copy = c1_slave_indices;
    MatrixData<T> c1_constraint_data( c1_constraint, c1_slave_indices, master_indices );
    MatrixData<T> vertices_constraint_data( vertices_constraint, c1_slave_indices_copy_copy, vertices_indices );
    MatrixData<T> c0_c1constraint_data( c0_c1_constraint, c1_slave_indices_copy, c0_slave_indices );
    auto c0_slave_c1_constraint_data = c0_c1constraint_data * poisson_ptr->ConstraintData();
    auto c0_slave_c1_vertices_constraint_data = c0_c1constraint_data * poisson_ptr->VerticesConstraintData();
    this->_constraintData = c1_constraint_data + c0_slave_c1_constraint_data + poisson_ptr->ConstraintData();
    _slaveMasterConstraintData =
        vertices_constraint_data + c0_slave_c1_vertices_constraint_data + poisson_ptr->VerticesConstraintData();
}

template <typename T>
class KLShellC1InterfaceVisitor : public BiharmonicInterfaceVisitor<3, T>
{
public:
    using Knot = typename BiharmonicInterfaceVisitor<3, T>::Knot;
    using Quadrature = typename BiharmonicInterfaceVisitor<3, T>::Quadrature;
    using QuadList = typename BiharmonicInterfaceVisitor<3, T>::QuadList;
    using KnotSpan = typename BiharmonicInterfaceVisitor<3, T>::KnotSpan;
    using KnotSpanlist = typename BiharmonicInterfaceVisitor<3, T>::KnotSpanlist;
    using LoadFunctor = typename BiharmonicInterfaceVisitor<3, T>::LoadFunctor;
    using Matrix = typename BiharmonicInterfaceVisitor<3, T>::Matrix;
    using Vector = typename BiharmonicInterfaceVisitor<3, T>::Vector;
    using DomainShared_ptr = typename BiharmonicInterfaceVisitor<3, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename BiharmonicInterfaceVisitor<3, T>::ConstraintIntegralElementAssembler;

    using BiharmonicInterfaceVisitor<3, T>::C1IntegralElementAssembler;

private:
    struct SlaveMasterAndAngle
    {
        SlaveMasterAndAngle()
        {
        }

        SlaveMasterAndAngle( const Vector& s, const Vector& m, const T angle )
            : _slaveQuadrature( s ), _masterQuadrature( m ), _angle( angle )
        {
        }
        Vector _slaveQuadrature;
        Vector _masterQuadrature;
        T _angle;
    };

public:
    KLShellC1InterfaceVisitor() : BiharmonicInterfaceVisitor<3, T>( std::make_unique<KLShellC0InterfaceVisitor<T>>() )
    {
    }
    void C1IntegralElementAssembler( Matrix& slave_constraint_basis,
                                     std::vector<int>& slave_constraint_basis_indices,
                                     Matrix& master_constraint_basis,
                                     std::vector<int>& master_constraint_basis_indices,
                                     Matrix& multiplier_basis,
                                     std::vector<int>& multiplier_basis_indices,
                                     T& integral_weight,
                                     Edge<3, T>* edge,
                                     const Quadrature& u )
    {
        auto multiplier_domain = edge->GetDomain();
        auto slave_domain = edge->Parent( 0 ).lock()->GetDomain();
        auto master_domain = edge->Counterpart().lock()->Parent( 0 ).lock()->GetDomain();

        //    set up integration weights
        integral_weight = u.second;

        // Map abscissa from Lagrange multiplier space to slave and master domain
        Vector slave_quadrature_abscissa, master_quadrature_abscissa;

        auto it = _quadratureMap.find( u.first( 0 ) );

        if ( it == _quadratureMap.end() )
        {
            if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa ) )
            {
                std::cout << "MapParametericPoint failed" << std::endl;
            }
            if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa ) )
            {
                std::cout << "MapParametericPoint failed" << std::endl;
            }
        }
        else
        {
            slave_quadrature_abscissa = ( it->second )._slaveQuadrature;
            master_quadrature_abscissa = ( it->second )._masterQuadrature;
        }

        // Evaluate derivative upto 1^st order in slave and master domain
        auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 1 );
        auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 1 );

        //  Evaluate Lagrange multiplier basis
        auto multiplier_evals = multiplier_domain->EvalDualAllTensor( u.first );

        // Resize integration matrices
        slave_constraint_basis.resize( 1, slave_evals->size() );
        multiplier_basis.resize( 1, multiplier_evals->size() );

        // Compute the following matrix
        // +-----------+-----------+
        // | ∂ξ_m/∂ξ_s | ∂η_m/∂ξ_s |
        // +-----------+-----------+
        // | ∂ξ_m/∂η_s | ∂η_m/∂η_s |
        // +-----------+-----------+
        Matrix slave_jacobian = slave_domain->JacobianMatrix( slave_quadrature_abscissa );
        Matrix master_jacobian = master_domain->JacobianMatrix( master_quadrature_abscissa );
        // Matrix master_to_slave = slave_jacobian * master_jacobian.inverse();
        Eigen::Matrix<T, 3, 1> A1_s, A2_s, A1_m, A2_m;

        A1_m = master_jacobian.col( 0 );
        A2_m = master_jacobian.col( 1 );

        switch ( edge->GetOrient() )
        {
        // For south and north edge derivative w.r.t η_s should be consistent
        case Orientation::south:
        case Orientation::north:
        {
            A1_s = slave_jacobian.col( 1 );
            A2_s = -slave_jacobian.col( 0 );
        }
        // For south and north edge derivative w.r.t ξ_s should be consistent
        case Orientation::east:
        case Orientation::west:
        {
            A1_s = slave_jacobian.col( 0 );
            A2_s = slave_jacobian.col( 1 );
        }
        }

        switch ( edge->Counterpart()->GetOrient() )
        {
        // For south and north edge derivative w.r.t η_s should be consistent
        case Orientation::south:
        case Orientation::north:
        {
            A2_s = A2_s.dot( A1_m ) * A1_m;
        }
        // For south and north edge derivative w.r.t ξ_s should be consistent
        case Orientation::east:
        case Orientation::west:
        {
            A2_s = A2_s.dot( A2_m ) * A2_m;
        }
        }

        const Eigen::Matrix<T, 3, 1> nA2_s = A2_s.normalized();
        T angle = 0;
        if ( it == _quadratureMap.end() )
        {
            Eigen::Matrix<T, 3, 1> A3_s = A1_s.cross( A2_s );
            Eigen::Matrix<T, 3, 1> A3_m = A1_m.cross( A2_m );
            A3_s.normalize();
            A3_m.normalize();
            auto sc = SinCosBetweenTwoUniVec( A3_m, A3_s, nA2_s );
            angle = sc.first >= 0 ? std::acos( sc.second ) : ( 2 * M_PI - std::acos( sc.second ) );
            _quadratureMap[u.first( 0 )] = SlaveMasterAndAngle( slave_quadrature_abscissa, master_quadrature_abscissa, angle );
        }
        else
        {
            angle = ( it->second )._angle;
        }

        const Eigen::Matrix<T, 3, 3> rotation_from_s_to_m =
            Accessory::RotationMatrix( nA2_s, -std::sin( angle ), std::cos( angle ) );

        Eigen::Matrix<T, 3, 1> A1_m_prime, A2_m_prime;
        A1_m_prime = rotation_from_s_to_m * A1_s;
        A2_m_prime = rotation_from_s_to_m * A2_s;

        Matrix gramian = master_jacobian.transpose() * master_jacobian;
        Vector A1_m_jac = gramian.partialPivLu().solve( master_jacobian.transpose() * A1_m_prime );
        Vector A2_m_jac = gramian.partialPivLu().solve( master_jacobian.transpose() * A2_m_prime );

        switch ( edge->GetOrient() )
        {
        // For south and north edge derivative w.r.t η_s should be consistent
        case Orientation::south:
        case Orientation::north:
        {
            for ( int j = 0; j < slave_evals->size(); ++j )
            {
                slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[2];
            }
            break;
        }
        // For south and north edge derivative w.r.t ξ_s should be consistent
        case Orientation::east:
        case Orientation::west:
        {
            for ( int j = 0; j < slave_evals->size(); ++j )
            {
                slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[1];
            }
            break;
        }
        }
        Matrix u1_m = A1_m_jac.transpose() * ( *master_evals ).bottomRows( 2 );
        Matrix u2_m = A2_m_jac.transpose() * ( *master_evals ).bottomRows( 2 );

        Eigen::Matrix<T, 3, 3> rotation_from_m_to_s = Accessory::RotationMatrix( nA2_s, std::sin( angle ), std::cos( angle ) );
        Eigen::Matrix<T, 3, 3> identity;
        identity.setIdentity();

        slave_constraint_basis = kroneckerProduct( slave_constraint_basis, identity ).eval();
        multiplier_basis = kroneckerProduct( multiplier_basis, identity ).eval();
        u1_m = kroneckerProduct( u1_m, identity ).eval();
        u2_m = kroneckerProduct( u2_m, identity ).eval();
        master_constraint_basis = rotation_from_m_to_s * u1_m + DRdotA1m( A1_m_prime, A2_m_prime, angle ) * u2_m;
    }

    Eigen::Matrix<T, 3, 3> DRdotA1m( const Vector& A1, const Vector& A2, const T angle )
    {
        Eigen::Matrix<T, 3, 3> res;
        const T A10 = A1( 0 ), A11 = A1( 1 ), A12 = A1( 2 );
        const T A20 = A2( 0 ), A21 = A2( 1 ), A22 = A2( 2 );
        const T A2_norm_square = std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 );
        res( 0, 0 ) = 2 * sin( angle / 2 ) *
                      ( A20 * ( A11 * A22 - A12 * A21 ) * std::sqrt( A2_norm_square ) * cos( angle / 2 ) +
                        ( 2 * A10 * A20 * ( A2_norm_square - std::pow( A20, 2 ) ) +
                          ( A11 * A21 + A12 * A22 ) * ( A2_norm_square - 2 * std::pow( A20, 2 ) ) ) *
                            sin( angle / 2 ) ) /
                      std::pow( A2_norm_square, 2 );
        res( 0, 1 ) =
            ( 2 * sin( angle / 2 ) *
              ( std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) *
                    ( A11 * A21 * A22 + A12 * ( std::pow( A20, 2 ) + std::pow( A22, 2 ) ) ) * cos( angle / 2 ) +
                A20 * ( -2 * A21 * ( A10 * A20 + A12 * A22 ) + A11 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) *
                    sin( angle / 2 ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 0, 2 ) = ( ( A20 * ( -2 * ( A10 * A20 + A11 * A21 ) * A22 +
                                  A12 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) - std::pow( A22, 2 ) ) ) +
                          A20 *
                              ( -( A12 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) ) ) +
                                2 * ( A10 * A20 + A11 * A21 ) * A22 + A12 * std::pow( A22, 2 ) ) *
                              cos( angle ) -
                          ( A11 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) ) + A12 * A21 * A22 ) *
                              std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) * sin( angle ) ) ) /
                      std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 1, 0 ) =
            ( ( A21 * ( -2 * A20 * ( A11 * A21 + A12 * A22 ) + A10 * ( -std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) +
                A21 * ( 2 * A20 * ( A11 * A21 + A12 * A22 ) + A10 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) - std::pow( A22, 2 ) ) ) *
                    cos( angle ) -
                std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) *
                    ( A10 * A20 * A22 + A12 * ( std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) * sin( angle ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 1, 1 ) = ( 2 * sin( angle / 2. ) *
                        ( A21 * ( A12 * A20 - A10 * A22 ) *
                              std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) * cos( angle / 2. ) +
                          ( 2 * A11 * A21 * ( std::pow( A20, 2 ) + std::pow( A22, 2 ) ) +
                            A10 * A20 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) + std::pow( A22, 2 ) ) +
                            A12 * A22 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) *
                              sin( angle / 2. ) ) ) /
                      std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 1, 2 ) =
            ( 2 * sin( angle / 2. ) *
              ( ( A10 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) ) + A12 * A20 * A22 ) *
                    std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) * cos( angle / 2. ) +
                A21 * ( -2 * ( A10 * A20 + A11 * A21 ) * A22 + A12 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) - std::pow( A22, 2 ) ) ) *
                    sin( angle / 2. ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 2, 0 ) =
            ( ( A22 * ( -2 * A20 * ( A11 * A21 + A12 * A22 ) + A10 * ( -std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) +
                A22 * ( 2 * A20 * ( A11 * A21 + A12 * A22 ) + A10 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) - std::pow( A22, 2 ) ) ) *
                    cos( angle ) +
                std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) *
                    ( A10 * A20 * A21 + A11 * ( std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) * sin( angle ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 2, 1 ) =
            ( ( A22 * ( -2 * A21 * ( A10 * A20 + A12 * A22 ) + A11 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) +
                A22 * ( 2 * A21 * ( A10 * A20 + A12 * A22 ) - A11 * ( std::pow( A20, 2 ) - std::pow( A21, 2 ) + std::pow( A22, 2 ) ) ) *
                    cos( angle ) -
                std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) *
                    ( A11 * A20 * A21 + A10 * ( std::pow( A20, 2 ) + std::pow( A22, 2 ) ) ) * sin( angle ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        res( 2, 2 ) =
            ( 2 * sin( angle / 2. ) *
              ( ( -( A11 * A20 ) + A10 * A21 ) * A22 *
                    std::sqrt( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ) ) * cos( angle / 2. ) +
                ( ( A10 * A20 + A11 * A21 ) * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) ) +
                  2 * A12 * ( std::pow( A20, 2 ) + std::pow( A21, 2 ) ) * A22 - ( A10 * A20 + A11 * A21 ) * std::pow( A22, 2 ) ) *
                    sin( angle / 2. ) ) ) /
            std::pow( std::pow( A20, 2 ) + std::pow( A21, 2 ) + std::pow( A22, 2 ), 2 );

        return res;
    }

    void SolveConstraint( Edge<3, T>* edge )
    {
        const int dimension = 3;
        auto vertices_indices = edge->VerticesIndices( dimension, 1 );

        // iterate across the constraint equation container and obtain activated local indices and lagrange multiplier indices
        std::vector<int> slave_indices = Accessory::ColIndicesVector( this->_c1Slave );
        std::vector<int> master_indices = Accessory::ColIndicesVector( this->_c1Master );
        std::vector<int> multiplier_indices = Accessory::RowIndicesVector( this->_c1Slave );

        auto poisson_ptr = dynamic_cast<KLShellC0InterfaceVisitor<T>*>( this->_poisson.get() );
        std::vector<int> c0_slave_indices = *( poisson_ptr->ConstraintData()._rowIndices );
        std::vector<int> c1_slave_complement_indices;
        std::set_union( vertices_indices.begin(), vertices_indices.end(), c0_slave_indices.begin(),
                        c0_slave_indices.end(), std::back_inserter( c1_slave_complement_indices ) );
        std::vector<int> c1_slave_indices;
        std::set_difference( slave_indices.begin(), slave_indices.end(), c1_slave_complement_indices.begin(),
                             c1_slave_complement_indices.end(), std::back_inserter( c1_slave_indices ) );

        auto c1_slave_indices_inverse_map = Accessory::IndicesInverseMap( c1_slave_indices );
        auto c0_slave_indices_inverse_map = Accessory::IndicesInverseMap( c0_slave_indices );
        auto master_indices_inverse_map = Accessory::IndicesInverseMap( master_indices );
        auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap( multiplier_indices );
        auto vertices_indices_inverse_map = Accessory::IndicesInverseMap( vertices_indices );

        std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs, condensed_c0_slave, condensed_vertices_rhs;
        this->CondensedTripletVia( multiplier_indices_inverse_map, c1_slave_indices_inverse_map, this->_c1Slave, condensed_gramian );
        this->CondensedTripletVia( multiplier_indices_inverse_map, master_indices_inverse_map, this->_c1Master, condensed_rhs );
        this->CondensedTripletVia( multiplier_indices_inverse_map, c0_slave_indices_inverse_map, this->_c1Slave, condensed_c0_slave );
        this->CondensedTripletVia( multiplier_indices_inverse_map, vertices_indices_inverse_map, this->_c1Slave, condensed_vertices_rhs );

        Matrix gramian_matrix, rhs_matrix, c0_slave_matrix, vertices_rhs_matrix;
        this->MatrixAssembler( multiplier_indices_inverse_map.size(), c1_slave_indices_inverse_map.size(),
                               condensed_gramian, gramian_matrix );
        this->MatrixAssembler( multiplier_indices_inverse_map.size(), master_indices_inverse_map.size(), condensed_rhs, rhs_matrix );
        this->MatrixAssembler( multiplier_indices_inverse_map.size(), c0_slave_indices_inverse_map.size(),
                               condensed_c0_slave, c0_slave_matrix );
        this->MatrixAssembler( multiplier_indices_inverse_map.size(), vertices_indices_inverse_map.size(),
                               condensed_vertices_rhs, vertices_rhs_matrix );

        Matrix c1_constraint = this->SolveNonSymmetric( gramian_matrix, rhs_matrix );
        Matrix c0_c1_constraint = this->SolveNonSymmetric( gramian_matrix, -c0_slave_matrix );
        Matrix vertices_constraint = this->SolveNonSymmetric( gramian_matrix, -vertices_rhs_matrix );
        auto c1_slave_indices_copy = c1_slave_indices;
        auto c1_slave_indices_copy_copy = c1_slave_indices;
        MatrixData<T> c1_constraint_data( c1_constraint, c1_slave_indices, master_indices );
        MatrixData<T> vertices_constraint_data( vertices_constraint, c1_slave_indices_copy_copy, vertices_indices );
        MatrixData<T> c0_c1constraint_data( c0_c1_constraint, c1_slave_indices_copy, c0_slave_indices );

        auto c0_slave_c1_constraint_data = c0_c1constraint_data * poisson_ptr->ConstraintData();
        auto c0_slave_c1_vertices_constraint_data = c0_c1constraint_data * poisson_ptr->VerticesConstraintData();
        this->_constraintData = c1_constraint_data + c0_slave_c1_constraint_data + poisson_ptr->ConstraintData();
        _slaveMasterConstraintData =
            vertices_constraint_data + c0_slave_c1_vertices_constraint_data + poisson_ptr->VerticesConstraintData();
        this->_constraintData.Print();
    }

protected:
    MatrixData<T> _slaveMasterConstraintData;
    std::unordered_map<T, SlaveMasterAndAngle> _quadratureMap;
};