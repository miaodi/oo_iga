//
// Created by di miao on 10/24/17.
//

#pragma once

#include "InterfaceVisitor.hpp"

template <int N, typename T>
class PoissonInterfaceVisitor : public InterfaceVisitor<N, T>
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
    PoissonInterfaceVisitor() : InterfaceVisitor<N, T>()
    {
    }

protected:
    void SolveConstraint( Edge<N, T>* );

    void LocalAssemble( Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan& );

    template <int n = N>
    typename std::enable_if<n == 3, void>::type C0IntegralElementAssembler( Matrix& slave_constraint_basis,
                                                                            std::vector<int>& slave_constraint_basis_indices,
                                                                            Matrix& master_constrint_basis,
                                                                            std::vector<int>& master_constraint_basis_indices,
                                                                            Matrix& multiplier_basis,
                                                                            std::vector<int>& multiplier_basis_indices,
                                                                            T& integral_weight,
                                                                            Edge<N, T>* edge,
                                                                            const Quadrature& u );

    template <int n = N>
    typename std::enable_if<n == 2, void>::type C0IntegralElementAssembler( Matrix& slave_constraint_basis,
                                                                            std::vector<int>& slave_constraint_basis_indices,
                                                                            Matrix& master_constrint_basis,
                                                                            std::vector<int>& master_constraint_basis_indices,
                                                                            Matrix& multiplier_basis,
                                                                            std::vector<int>& multiplier_basis_indices,
                                                                            T& integral_weight,
                                                                            Edge<N, T>* edge,
                                                                            const Quadrature& u );

protected:
    std::vector<Eigen::Triplet<T>> _c0Slave;
    std::vector<Eigen::Triplet<T>> _c0Master;
};

template <int N, typename T>
void PoissonInterfaceVisitor<N, T>::SolveConstraint( Edge<N, T>* edge )
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> slave_activated_indices = Accessory::ColIndicesVector( _c0Slave );
    std::vector<int> master_activated_indices = Accessory::ColIndicesVector( _c0Master );
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector( _c0Slave );
    auto activated_slave_indices_inverse_map = Accessory::IndicesInverseMap( slave_activated_indices );
    auto activated_master_indices_inverse_map = Accessory::IndicesInverseMap( master_activated_indices );
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap( multiplier_indices );
    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs;
    this->CondensedTripletVia( multiplier_indices_inverse_map, activated_slave_indices_inverse_map, _c0Slave, condensed_gramian );
    this->CondensedTripletVia( multiplier_indices_inverse_map, activated_master_indices_inverse_map, _c0Master, condensed_rhs );
    Matrix gramian_matrix, rhs_matrix;
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), activated_slave_indices_inverse_map.size(),
                           condensed_gramian, gramian_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), activated_master_indices_inverse_map.size(),
                           condensed_rhs, rhs_matrix );
    Accessory::removeNoise( gramian_matrix, 1e-7 * abs( gramian_matrix( 0, 0 ) ) );
    Accessory::removeNoise( rhs_matrix, 1e-14 );
    Matrix constraint = this->SolveNonSymmetric( gramian_matrix, rhs_matrix );
    MatrixData<T> constraint_data( constraint, slave_activated_indices, master_activated_indices );
    this->_constraintData = std::move( constraint_data );
}

template <int N, typename T>
void PoissonInterfaceVisitor<N, T>::LocalAssemble( Element<1, N, T>* g, const QuadratureRule<T>& quadrature_rule, const KnotSpan& knot_span )
{
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto c0_function =
        std::bind( &PoissonInterfaceVisitor<N, T>::C0IntegralElementAssembler<>, this, _1, _2, _3, _4, _5, _6, _7, _8, _9 );
    this->ConstraintLocalAssemble( g, quadrature_rule, knot_span, c0_function, _c0Slave, _c0Master );
}

template <int N, typename T>
template <int n>
typename std::enable_if<n == 3, void>::type PoissonInterfaceVisitor<N, T>::C0IntegralElementAssembler(
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

    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 0 );
    auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 0 );
    auto multiplier_evals = multiplier_domain->EvalDualAllTensor( u.first );

    slave_constraint_basis.resize( 1, slave_evals->size() );
    master_constraint_basis.resize( 1, master_evals->size() );
    multiplier_basis.resize( 1, multiplier_evals->size() );

    for ( int j = 0; j < slave_evals->size(); ++j )
    {
        slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[0];
    }
    for ( int j = 0; j < master_evals->size(); ++j )
    {
        master_constraint_basis( 0, j ) = ( *master_evals )[j].second[0];
    }
    for ( int j = 0; j < multiplier_evals->size(); ++j )
    {
        multiplier_basis( 0, j ) = ( *multiplier_evals )[j].second[0];
    }

    Eigen::Matrix<T, 3, 3> identity;
    identity.setIdentity();

    master_constraint_basis = kroneckerProduct( master_constraint_basis, identity ).eval();
    slave_constraint_basis = kroneckerProduct( slave_constraint_basis, identity ).eval();
    multiplier_basis = kroneckerProduct( multiplier_basis, identity ).eval();

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
typename std::enable_if<n == 2, void>::type PoissonInterfaceVisitor<N, T>::C0IntegralElementAssembler(
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

    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if ( !Accessory::MapParametricPoint( &*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa ) )
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    auto slave_evals = slave_domain->EvalDerAllTensor( slave_quadrature_abscissa, 0 );
    auto master_evals = master_domain->EvalDerAllTensor( master_quadrature_abscissa, 0 );
    // auto multiplier_evals = multiplier_domain->EvalDualAllTensor( u.first );
    auto multiplier_evals = ( multiplier_domain->BasisGetter( 0 ) ).EvalModifiedDerAll( u.first( 0 ), 0 );

    slave_constraint_basis.resize( 1, slave_evals->size() );
    master_constraint_basis.resize( 1, master_evals->size() );
    multiplier_basis.resize( 1, multiplier_evals->size() );

    for ( int j = 0; j < slave_evals->size(); ++j )
    {
        slave_constraint_basis( 0, j ) = ( *slave_evals )[j].second[0];
    }
    for ( int j = 0; j < master_evals->size(); ++j )
    {
        master_constraint_basis( 0, j ) = ( *master_evals )[j].second[0];
    }
    for ( int j = 0; j < multiplier_evals->size(); ++j )
    {
        multiplier_basis( 0, j ) = ( *multiplier_evals )[j].second[0];
    }

    // set up local indices corresponding to test basis functions and trial basis functions
    if ( slave_constraint_basis_indices.size() == 0 )
    {
        auto index = slave_domain->ActiveIndex( slave_quadrature_abscissa );
        for ( auto& i : index )
        {
            slave_constraint_basis_indices.push_back( i );
        }
    }
    if ( master_constraint_basis_indices.size() == 0 )
    {
        auto index = master_domain->ActiveIndex( master_quadrature_abscissa );
        for ( auto& i : index )
        {
            master_constraint_basis_indices.push_back( i );
        }
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
class PoissonCodimensionInterfaceVisitor : public PoissonInterfaceVisitor<N, T>
{
public:
    using Knot = typename PoissonInterfaceVisitor<N, T>::Knot;
    using Quadrature = typename PoissonInterfaceVisitor<N, T>::Quadrature;
    using QuadList = typename PoissonInterfaceVisitor<N, T>::QuadList;
    using KnotSpan = typename PoissonInterfaceVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename PoissonInterfaceVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename PoissonInterfaceVisitor<N, T>::LoadFunctor;
    using Matrix = typename PoissonInterfaceVisitor<N, T>::Matrix;
    using Vector = typename PoissonInterfaceVisitor<N, T>::Vector;
    using DomainShared_ptr = typename PoissonInterfaceVisitor<N, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename PoissonInterfaceVisitor<N, T>::ConstraintIntegralElementAssembler;

public:
    PoissonCodimensionInterfaceVisitor( const int& c ) : PoissonInterfaceVisitor<N, T>(), codimension{c}
    {
    }

    const MatrixData<T>& VerticesConstraintData() const
    {
        return _slaveMasterConstraintData;
    }

protected:
    void SolveConstraint( Edge<N, T>* );

protected:
    int codimension;
    MatrixData<T> _slaveMasterConstraintData;
};

template <int N, typename T>
void PoissonCodimensionInterfaceVisitor<N, T>::SolveConstraint( Edge<N, T>* edge )
{
    const int dimension = 1;
    auto vertices_indices = edge->VerticesIndices( dimension, codimension - 1 );

    // iterate across the constraint equation container and obtain activated local indices and lagrange multiplier indices
    std::vector<int> slave_side_activated_indices = Accessory::ColIndicesVector( this->_c0Slave );
    std::vector<int> master_activated_indices = Accessory::ColIndicesVector( this->_c0Master );
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector( this->_c0Slave );

    // filter master dof from slave side due to codimension
    std::vector<int> slave_activated_indices, vertices_activated_indices, tmp_vector;
    Accessory::decompose_sets( slave_side_activated_indices.begin(), slave_side_activated_indices.end(),
                               vertices_indices.begin(), vertices_indices.end(), back_inserter( slave_activated_indices ),
                               back_inserter( tmp_vector ), back_inserter( vertices_activated_indices ) );

    auto activated_slave_indices_inverse_map = Accessory::IndicesInverseMap( slave_activated_indices );
    auto activated_master_indices_inverse_map = Accessory::IndicesInverseMap( master_activated_indices );
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap( multiplier_indices );
    auto vertices_indices_inverse_map = Accessory::IndicesInverseMap( vertices_activated_indices );

    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs, condensed_viertices_rhs;
    this->CondensedTripletVia( multiplier_indices_inverse_map, activated_slave_indices_inverse_map, this->_c0Slave, condensed_gramian );
    this->CondensedTripletVia( multiplier_indices_inverse_map, vertices_indices_inverse_map, this->_c0Slave, condensed_viertices_rhs );
    this->CondensedTripletVia( multiplier_indices_inverse_map, activated_master_indices_inverse_map, this->_c0Master, condensed_rhs );
    Matrix gramian_matrix, rhs_matrix, vertices_rhs_matrix;
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), activated_slave_indices_inverse_map.size(),
                           condensed_gramian, gramian_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), vertices_indices_inverse_map.size(),
                           condensed_viertices_rhs, vertices_rhs_matrix );
    this->MatrixAssembler( multiplier_indices_inverse_map.size(), activated_master_indices_inverse_map.size(),
                           condensed_rhs, rhs_matrix );

    // Accessory::removeNoise( gramian_matrix, 1e-7 * abs( vertices_rhs_matrix( 0, 0 ) ) );
    // Accessory::removeNoise( rhs_matrix, 1e-14 );
    // Accessory::removeNoise( vertices_rhs_matrix, 1e-7 * abs( vertices_rhs_matrix( 0, 0 ) ) );

    Matrix constraint = this->SolveNonSymmetric( gramian_matrix, rhs_matrix );
    Matrix vertices_constraint = this->SolveNonSymmetric( -gramian_matrix, vertices_rhs_matrix );

    auto slave_activated_indices1 = slave_activated_indices;
    MatrixData<T> constraint_data( constraint, slave_activated_indices, master_activated_indices );
    MatrixData<T> vertices_constraint_data( vertices_constraint, slave_activated_indices1, vertices_activated_indices );
    this->_constraintData = std::move( constraint_data );
    _slaveMasterConstraintData = std::move( vertices_constraint_data );
}