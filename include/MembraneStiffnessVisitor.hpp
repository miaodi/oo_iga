//
// Created by miaodi on 30/11/2017.
//

#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class MembraneStiffnessVisitor : public StiffnessVisitor<3, 3, T>
{
public:
    using Knot = typename StiffnessVisitor<3, 3, T>::Knot;
    using Quadrature = typename StiffnessVisitor<3, 3, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<3, 3, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<3, 3, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<3, 3, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<3, 3, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<3, 3, T>::Matrix;
    using Vector = typename StiffnessVisitor<3, 3, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<3, 3, T>::DomainShared_ptr;
    using DataType = T;

public:
    MembraneStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<3, 3, T>( body_force )
    {
    }

    void SetStateDatas( T* disp, T* vel )
    {
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const;

protected:
    T _nu{.0};
    T _E{4.32e8};
    T _h{0.25};

    // T _nu{.3};
    // T _E{6.825e7};
    // T _h{0.04};
};

template <typename T>
void MembraneStiffnessVisitor<T>::IntegralElementAssembler( Matrix& bilinear_form_trail,
                                                            Matrix& bilinear_form_test,
                                                            Matrix& linear_form_value,
                                                            Matrix& linear_form_test,
                                                            const DomainShared_ptr domain,
                                                            const Knot& u ) const
{
    auto evals = domain->EvalDerAllTensor( u, 1 );
    linear_form_value.resize( 3, 1 );
    linear_form_value( 0, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[0];
    linear_form_value( 1, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[1];
    linear_form_value( 2, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[2];

    linear_form_test.resize( 3, 3 * evals->size() );
    linear_form_test.setZero();
    bilinear_form_trail.resize( 3, 3 * evals->size() );
    bilinear_form_trail.setZero();

    Eigen::Matrix<T, 3, 1> u1, u2, u3, v1, v2, v3;
    u1 = domain->AffineMap( u, {1, 0} );
    u2 = domain->AffineMap( u, {0, 1} );
    u3 = u1.cross( u2 );
    T jacobian = u3.norm();
    u3 *= 1.0 / jacobian;

    std::tie( v1, v2, v3 ) = Accessory::CovariantToContravariant( u1, u2, u3 );

    for ( int j = 0; j < evals->size(); ++j )
    {
        linear_form_test( 0, 3 * j ) = ( *evals )[j].second[0];
        linear_form_test( 1, 3 * j + 1 ) = ( *evals )[j].second[0];
        linear_form_test( 2, 3 * j + 2 ) = ( *evals )[j].second[0];

        bilinear_form_trail( 0, 3 * j ) = ( *evals )[j].second[1] * u1( 0 );
        bilinear_form_trail( 0, 3 * j + 1 ) = ( *evals )[j].second[1] * u1( 1 );
        bilinear_form_trail( 0, 3 * j + 2 ) = ( *evals )[j].second[1] * u1( 2 );

        bilinear_form_trail( 1, 3 * j ) = ( *evals )[j].second[2] * u2( 0 );
        bilinear_form_trail( 1, 3 * j + 1 ) = ( *evals )[j].second[2] * u2( 1 );
        bilinear_form_trail( 1, 3 * j + 2 ) = ( *evals )[j].second[2] * u2( 2 );

        bilinear_form_trail( 2, 3 * j ) = ( *evals )[j].second[1] * u2( 0 ) + ( *evals )[j].second[2] * u1( 0 );
        bilinear_form_trail( 2, 3 * j + 1 ) = ( *evals )[j].second[1] * u2( 1 ) + ( *evals )[j].second[2] * u1( 1 );
        bilinear_form_trail( 2, 3 * j + 2 ) = ( *evals )[j].second[1] * u2( 2 ) + ( *evals )[j].second[2] * u1( 2 );
    }

    T v11, v12, v22;
    v11 = v1.dot( v1 );
    v22 = v2.dot( v2 );
    v12 = v1.dot( v2 );

    Matrix H( 3, 3 );
    H << v11 * v11, _nu * v11 * v22 + ( 1 - _nu ) * v12 * v12, v11 * v12, _nu * v11 * v22 + ( 1 - _nu ) * v12 * v12,
        v22 * v22, v22 * v12, v11 * v12, v22 * v12, .5 * ( ( 1 - _nu ) * v11 * v22 + ( 1 + _nu ) * v12 * v12 );
    bilinear_form_test = _E * _h / ( 1 - _nu * _nu ) * H * bilinear_form_trail;
}

enum class NonlinearMembraneStiffnessType
{
    Default = -1, // dn:depsilon
    First = 0,    // alpha=1, beta=1
    Second = 1,   // alpha=2, beta=2
    Third = 2,    // alpha=1, beta=2
    Fourth = 3,   // all term except default
};

template <typename T, NonlinearMembraneStiffnessType nt>
class NonlinearMembraneStiffnessVisitor : public StiffnessVisitor<3, 3, T>
{
public:
    using Knot = typename StiffnessVisitor<3, 3, T>::Knot;
    using Quadrature = typename StiffnessVisitor<3, 3, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<3, 3, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<3, 3, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<3, 3, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<3, 3, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<3, 3, T>::Matrix;
    using Vector = typename StiffnessVisitor<3, 3, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<3, 3, T>::DomainShared_ptr;
    using DataType = T;

public:
    NonlinearMembraneStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<3, 3, T>( body_force )
    {
    }

    void SetStateDatas( T* disp, T* vel )
    {
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const
    {
        Assemble( bilinear_form_trail, bilinear_form_test, linear_form_value, linear_form_test, domain, u );
    }

    template <NonlinearMembraneStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearMembraneStiffnessType::Default, void>::type Assemble( Matrix& bilinear_form_trail,
                                                                                                 Matrix& bilinear_form_test,
                                                                                                 Matrix& linear_form_value,
                                                                                                 Matrix& linear_form_test,
                                                                                                 const DomainShared_ptr domain,
                                                                                                 const Knot& u ) const;

    template <NonlinearMembraneStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearMembraneStiffnessType::First, void>::type Assemble( Matrix& bilinear_form_trail,
                                                                                               Matrix& bilinear_form_test,
                                                                                               Matrix& linear_form_value,
                                                                                               Matrix& linear_form_test,
                                                                                               const DomainShared_ptr domain,
                                                                                               const Knot& u ) const;

    template <NonlinearMembraneStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearMembraneStiffnessType::Second, void>::type Assemble( Matrix& bilinear_form_trail,
                                                                                                Matrix& bilinear_form_test,
                                                                                                Matrix& linear_form_value,
                                                                                                Matrix& linear_form_test,
                                                                                                const DomainShared_ptr domain,
                                                                                                const Knot& u ) const;

    template <NonlinearMembraneStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearMembraneStiffnessType::Third, void>::type Assemble( Matrix& bilinear_form_trail,
                                                                                               Matrix& bilinear_form_test,
                                                                                               Matrix& linear_form_value,
                                                                                               Matrix& linear_form_test,
                                                                                               const DomainShared_ptr domain,
                                                                                               const Knot& u ) const;

    template <NonlinearMembraneStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearMembraneStiffnessType::Fourth, void>::type Assemble( Matrix& bilinear_form_trail,
                                                                                                Matrix& bilinear_form_test,
                                                                                                Matrix& linear_form_value,
                                                                                                Matrix& linear_form_test,
                                                                                                const DomainShared_ptr domain,
                                                                                                const Knot& u ) const;

protected:
    // T _nu{.0};
    // T _E{4.32e8};
    // T _h{0.25};

    T _nu{.0};
    T _E{21e6};
    T _h{0.03};

    // T _nu{.0};
    // T _E{1.2e6};
    // T _h{0.1};

    // T _nu{.3};
    // T _E{6.825e7};
    // T _h{0.04};
};

template <typename T, NonlinearMembraneStiffnessType nt>
template <NonlinearMembraneStiffnessType tt>
typename std::enable_if<tt == NonlinearMembraneStiffnessType::Default, void>::type NonlinearMembraneStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trail,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 3, 1> U1, U2, U3, V1, V2, V3, u1, u2, u3;
    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U3 = U1.cross( U2 );
    const T jacobian = U3.norm();
    U3 *= 1.0 / jacobian;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );

    linear_form_value.resize( 6, 1 );
    linear_form_value( 0, 0 ) = .5 * ( u1.dot( u1 ) - U1.dot( U1 ) );
    linear_form_value( 1, 0 ) = .5 * ( u2.dot( u2 ) - U2.dot( U2 ) );
    linear_form_value( 2, 0 ) = 1.0 * ( u1.dot( u2 ) - U1.dot( U2 ) );
    linear_form_value( 3, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[0];
    linear_form_value( 4, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[1];
    linear_form_value( 5, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[2];

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    Matrix du1_Bmatrix, du2_Bmatrix, u_Bmatrix;
    du1_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du2_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    u_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du1_Bmatrix.setZero();
    du2_Bmatrix.setZero();
    u_Bmatrix.setZero();

    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        du1_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];

        du2_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        u_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[0];
        u_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[0];
        u_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[0];
    }

    bilinear_form_trail.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_trail.middleRows( 0, 1 ) = u1.transpose() * du1_Bmatrix;
    bilinear_form_trail.middleRows( 1, 1 ) = u2.transpose() * du2_Bmatrix;
    bilinear_form_trail.middleRows( 2, 1 ) = u1.transpose() * du2_Bmatrix + u2.transpose() * du1_Bmatrix;

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    bilinear_form_test = _E * _h / ( 1 - _nu * _nu ) * H * bilinear_form_trail;

    linear_form_test.resize( 6, 3 * current_config_evals->size() );
    linear_form_test.middleRows( 0, 3 ) = bilinear_form_test;
    linear_form_test.middleRows( 3, 3 ) = u_Bmatrix;
}

template <typename T, NonlinearMembraneStiffnessType nt>
template <NonlinearMembraneStiffnessType tt>
typename std::enable_if<tt == NonlinearMembraneStiffnessType::First, void>::type NonlinearMembraneStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trail,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 3, 1> U1, U2, U3, V1, V2, V3, u1, u2, u3;
    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U3 = U1.cross( U2 );
    const T jacobian = U3.norm();
    U3 *= 1.0 / jacobian;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );

    Vector epsilon( 3 );
    epsilon( 0 ) = .5 * ( u1.dot( u1 ) - U1.dot( U1 ) );
    epsilon( 1 ) = .5 * ( u2.dot( u2 ) - U2.dot( U2 ) );
    epsilon( 2 ) = 1.0 * ( u1.dot( u2 ) - U1.dot( U2 ) );

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    Vector n = _E * _h / ( 1 - _nu * _nu ) * H * epsilon;

    bilinear_form_trail.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_trail.setZero();
    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        bilinear_form_trail( 0, 3 * j ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];
    }
    bilinear_form_test = bilinear_form_trail * n( 0 );
    linear_form_value.resize( 1, 1 );
    linear_form_value.setZero();
    linear_form_test.resize( 1, 3 * current_config_evals->size() );
    linear_form_test.setZero();
}

template <typename T, NonlinearMembraneStiffnessType nt>
template <NonlinearMembraneStiffnessType tt>
typename std::enable_if<tt == NonlinearMembraneStiffnessType::Second, void>::type NonlinearMembraneStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trail,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 3, 1> U1, U2, U3, V1, V2, V3, u1, u2, u3;
    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U3 = U1.cross( U2 );
    const T jacobian = U3.norm();
    U3 *= 1.0 / jacobian;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );

    Vector epsilon( 3 );
    epsilon( 0 ) = .5 * ( u1.dot( u1 ) - U1.dot( U1 ) );
    epsilon( 1 ) = .5 * ( u2.dot( u2 ) - U2.dot( U2 ) );
    epsilon( 2 ) = 1.0 * ( u1.dot( u2 ) - U1.dot( U2 ) );

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    Vector n = _E * _h / ( 1 - _nu * _nu ) * H * epsilon;

    bilinear_form_trail.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_trail.setZero();
    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        bilinear_form_trail( 0, 3 * j ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];
    }
    bilinear_form_test = bilinear_form_trail * n( 1 );
    linear_form_value.resize( 1, 1 );
    linear_form_value.setZero();
    linear_form_test.resize( 1, 3 * current_config_evals->size() );
    linear_form_test.setZero();
}

template <typename T, NonlinearMembraneStiffnessType nt>
template <NonlinearMembraneStiffnessType tt>
typename std::enable_if<tt == NonlinearMembraneStiffnessType::Third, void>::type NonlinearMembraneStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trail,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 3, 1> U1, U2, U3, V1, V2, V3, u1, u2, u3;
    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U3 = U1.cross( U2 );
    const T jacobian = U3.norm();
    U3 *= 1.0 / jacobian;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );

    Vector epsilon( 3 );
    epsilon( 0 ) = .5 * ( u1.dot( u1 ) - U1.dot( U1 ) );
    epsilon( 1 ) = .5 * ( u2.dot( u2 ) - U2.dot( U2 ) );
    epsilon( 2 ) = 1.0 * ( u1.dot( u2 ) - U1.dot( U2 ) );

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    Vector n = _E * _h / ( 1 - _nu * _nu ) * H * epsilon;

    bilinear_form_trail.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_trail.setZero();
    bilinear_form_test.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_test.setZero();
    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        bilinear_form_trail( 0, 3 * j ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        bilinear_form_test( 0, 3 * j ) = ( *current_config_evals )[j].second[1] * n( 2 );
        bilinear_form_test( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1] * n( 2 );
        bilinear_form_test( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1] * n( 2 );
    }
    linear_form_value.resize( 1, 1 );
    linear_form_value.setZero();
    linear_form_test.resize( 1, 3 * current_config_evals->size() );
    linear_form_test.setZero();
}

template <typename T, NonlinearMembraneStiffnessType nt>
template <NonlinearMembraneStiffnessType tt>
typename std::enable_if<tt == NonlinearMembraneStiffnessType::Fourth, void>::type NonlinearMembraneStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trail,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 3, 1> U1, U2, U3, V1, V2, V3, u1, u2, u3;
    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U3 = U1.cross( U2 );
    const T jacobian = U3.norm();
    U3 *= 1.0 / jacobian;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );

    Vector epsilon( 3 );
    epsilon( 0 ) = .5 * ( u1.dot( u1 ) - U1.dot( U1 ) );
    epsilon( 1 ) = .5 * ( u2.dot( u2 ) - U2.dot( U2 ) );
    epsilon( 2 ) = 1.0 * ( u1.dot( u2 ) - U1.dot( U2 ) );

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    Vector n = _E * _h / ( 1 - _nu * _nu ) * H * epsilon;

    bilinear_form_trail.resize( 12, 3 * current_config_evals->size() );
    bilinear_form_trail.setZero();
    bilinear_form_test.resize( 12, 3 * current_config_evals->size() );
    bilinear_form_test.setZero();
    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        bilinear_form_trail( 0, 3 * j ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];

        bilinear_form_test( 0, 3 * j ) = ( *current_config_evals )[j].second[1] * n( 0 );
        bilinear_form_test( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1] * n( 0 );
        bilinear_form_test( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1] * n( 0 );

        bilinear_form_trail( 3, 3 * j ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 4, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 5, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        bilinear_form_test( 3, 3 * j ) = ( *current_config_evals )[j].second[2] * n( 1 );
        bilinear_form_test( 4, 3 * j + 1 ) = ( *current_config_evals )[j].second[2] * n( 1 );
        bilinear_form_test( 5, 3 * j + 2 ) = ( *current_config_evals )[j].second[2] * n( 1 );

        bilinear_form_trail( 6, 3 * j ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 7, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        bilinear_form_trail( 8, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        bilinear_form_test( 6, 3 * j ) = ( *current_config_evals )[j].second[1] * n( 2 );
        bilinear_form_test( 7, 3 * j + 1 ) = ( *current_config_evals )[j].second[1] * n( 2 );
        bilinear_form_test( 8, 3 * j + 2 ) = ( *current_config_evals )[j].second[1] * n( 2 );

        bilinear_form_trail( 9, 3 * j ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 10, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        bilinear_form_trail( 11, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];

        bilinear_form_test( 9, 3 * j ) = ( *current_config_evals )[j].second[2] * n( 2 );
        bilinear_form_test( 10, 3 * j + 1 ) = ( *current_config_evals )[j].second[2] * n( 2 );
        bilinear_form_test( 11, 3 * j + 2 ) = ( *current_config_evals )[j].second[2] * n( 2 );
    }
    linear_form_value.resize( 1, 1 );
    linear_form_value.setZero();
    linear_form_test.resize( 1, 3 * current_config_evals->size() );
    linear_form_test.setZero();
}
