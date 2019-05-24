//
// Created by miaodi on 30/11/2017.
//

#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class BendingStiffnessVisitor : public StiffnessVisitor<3, 3, T>
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
    BendingStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<3, 3, T>( body_force )
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
void BendingStiffnessVisitor<T>::IntegralElementAssembler( Matrix& bilinear_form_trail,
                                                           Matrix& bilinear_form_test,
                                                           Matrix& linear_form_value,
                                                           Matrix& linear_form_test,
                                                           const DomainShared_ptr domain,
                                                           const Knot& u ) const
{
    auto evals = domain->EvalDerAllTensor( u, 2 );
    linear_form_value.resize( 3, 1 );
    linear_form_value( 0, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[0];
    linear_form_value( 1, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[1];
    linear_form_value( 2, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[2];

    linear_form_test.resize( 3, 3 * evals->size() );
    linear_form_test.setZero();
    bilinear_form_trail.resize( 3, 3 * evals->size() );
    bilinear_form_trail.setZero();

    Eigen::Matrix<T, 3, 1> u1, u2, u3, u11, u12, u22, v1, v2, v3;

    u1 = domain->AffineMap( u, {1, 0} );
    u2 = domain->AffineMap( u, {0, 1} );
    u11 = domain->AffineMap( u, {2, 0} );
    u12 = domain->AffineMap( u, {1, 1} );
    u22 = domain->AffineMap( u, {0, 2} );
    u3 = u1.cross( u2 );
    T jacobian = u3.norm();
    u3 *= 1.0 / jacobian;

    std::tie( v1, v2, v3 ) = Accessory::CovariantToContravariant( u1, u2, u3 );

    for ( int j = 0; j < evals->size(); ++j )
    {
        linear_form_test( 0, 3 * j ) = ( *evals )[j].second[0];
        linear_form_test( 1, 3 * j + 1 ) = ( *evals )[j].second[0];
        linear_form_test( 2, 3 * j + 2 ) = ( *evals )[j].second[0];

        Eigen::Matrix<T, 3, 1> B1, B2, B3;
        B1 = -( *evals )[j].second[3] * u3 +
             1.0 / jacobian *
                 ( ( *evals )[j].second[1] * u11.cross( u2 ) + ( *evals )[j].second[2] * u1.cross( u11 ) +
                   u3.dot( u11 ) * ( ( *evals )[j].second[1] * u2.cross( u3 ) + ( *evals )[j].second[2] * u3.cross( u1 ) ) );
        B2 = -( *evals )[j].second[5] * u3 +
             1.0 / jacobian *
                 ( ( *evals )[j].second[1] * u22.cross( u2 ) + ( *evals )[j].second[2] * u1.cross( u22 ) +
                   u3.dot( u22 ) * ( ( *evals )[j].second[1] * u2.cross( u3 ) + ( *evals )[j].second[2] * u3.cross( u1 ) ) );
        B3 = 2 * ( -( *evals )[j].second[4] * u3 +
                   1.0 / jacobian *
                       ( ( *evals )[j].second[1] * u12.cross( u2 ) + ( *evals )[j].second[2] * u1.cross( u12 ) +
                         u3.dot( u12 ) * ( ( *evals )[j].second[1] * u2.cross( u3 ) + ( *evals )[j].second[2] * u3.cross( u1 ) ) ) );

        bilinear_form_trail( 0, 3 * j ) = B1( 0 );
        bilinear_form_trail( 0, 3 * j + 1 ) = B1( 1 );
        bilinear_form_trail( 0, 3 * j + 2 ) = B1( 2 );

        bilinear_form_trail( 1, 3 * j ) = B2( 0 );
        bilinear_form_trail( 1, 3 * j + 1 ) = B2( 1 );
        bilinear_form_trail( 1, 3 * j + 2 ) = B2( 2 );

        bilinear_form_trail( 2, 3 * j ) = B3( 0 );
        bilinear_form_trail( 2, 3 * j + 1 ) = B3( 1 );
        bilinear_form_trail( 2, 3 * j + 2 ) = B3( 2 );
    }

    T v11, v12, v22;
    v11 = v1.dot( v1 );
    v22 = v2.dot( v2 );
    v12 = v1.dot( v2 );

    Matrix H( 3, 3 );
    H << v11 * v11, _nu * v11 * v22 + ( 1 - _nu ) * v12 * v12, v11 * v12, _nu * v11 * v22 + ( 1 - _nu ) * v12 * v12,
        v22 * v22, v22 * v12, v11 * v12, v22 * v12, .5 * ( ( 1 - _nu ) * v11 * v22 + ( 1 + _nu ) * v12 * v12 );
    bilinear_form_test = _E * pow( _h, 3 ) / 12 / ( 1 - _nu * _nu ) * H * bilinear_form_trail;
}

enum class NonlinearBendingStiffnessType
{
    Default = -1, // dm:dkappa
    First = 0,    // alpha=1, beta=1
    Second = 1,   // alpha=2, beta=2
    Third = 2,

    AllExcpet = 3, // all term except default
};

template <typename T, NonlinearBendingStiffnessType nt>
class NonlinearBendingStiffnessVisitor : public StiffnessVisitor<3, 3, T>
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
    NonlinearBendingStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<3, 3, T>( body_force )
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

    template <NonlinearBendingStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearBendingStiffnessType::Default, void>::type Assemble( Matrix& bilinear_form_trial,
                                                                                                Matrix& bilinear_form_test,
                                                                                                Matrix& linear_form_value,
                                                                                                Matrix& linear_form_test,
                                                                                                const DomainShared_ptr domain,
                                                                                                const Knot& u ) const;

    template <NonlinearBendingStiffnessType tt = nt>
    typename std::enable_if<tt == NonlinearBendingStiffnessType::AllExcpet, void>::type Assemble( Matrix& bilinear_form_trial,
                                                                                                  Matrix& bilinear_form_test,
                                                                                                  Matrix& linear_form_value,
                                                                                                  Matrix& linear_form_test,
                                                                                                  const DomainShared_ptr domain,
                                                                                                  const Knot& u ) const;

protected:
    // T _nu{.0};
    // T _E{4.32e8};
    // T _h{0.25};

    // T _nu{.0};
    // T _E{1.2e6};
    // T _h{0.1};

    T _nu{.0};
    T _E{21e6};
    T _h{0.03};

    // T _nu{.3};
    // T _E{6.825e7};
    // T _h{0.04};
};

template <typename T, NonlinearBendingStiffnessType nt>
template <NonlinearBendingStiffnessType tt>
typename std::enable_if<tt == NonlinearBendingStiffnessType::Default, void>::type NonlinearBendingStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trial,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 2 );
    linear_form_value.resize( 3, 1 );
    bilinear_form_trial.resize( 3, 3 * current_config_evals->size() );
    bilinear_form_trial.setZero();

    Eigen::Matrix<T, 3, 1> U1, U2, U3, U11, U12, U22, V1, V2, V3, u1, u2, u3, u11, u12, u22;

    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U11 = domain->AffineMap( u, {2, 0} );
    U12 = domain->AffineMap( u, {1, 1} );
    U22 = domain->AffineMap( u, {0, 2} );
    U3 = U1.cross( U2 );
    const T jacobian_ref = U3.norm();
    U3 *= 1.0 / jacobian_ref;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );
    u11 = current_config.AffineMap( u, {2, 0} );
    u12 = current_config.AffineMap( u, {1, 1} );
    u22 = current_config.AffineMap( u, {0, 2} );
    u3 = u1.cross( u2 );
    const T jacobian_cur = u3.norm();
    u3 *= 1.0 / jacobian_cur;

    linear_form_value.resize( 3, 1 );
    linear_form_value( 0, 0 ) = U11.dot( U3 ) - u11.dot( u3 );
    linear_form_value( 1, 0 ) = U22.dot( U3 ) - u22.dot( u3 );
    linear_form_value( 2, 0 ) = 2 * ( U12.dot( U3 ) - u12.dot( u3 ) );

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    Matrix du1_Bmatrix, du2_Bmatrix, du11_Bmatrix, du22_Bmatrix, du12_Bmatrix;
    du1_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du2_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du11_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du22_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du12_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du1_Bmatrix.setZero();
    du2_Bmatrix.setZero();
    du11_Bmatrix.setZero();
    du22_Bmatrix.setZero();
    du12_Bmatrix.setZero();

    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        du1_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];

        du2_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        du11_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[3];
        du11_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[3];
        du11_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[3];

        du22_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[5];
        du22_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[5];
        du22_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[5];

        du12_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[4];
        du12_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[4];
        du12_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[4];
    }

    bilinear_form_trial.middleRows( 0, 1 ) =
        -u3.transpose() * du11_Bmatrix +
        1.0 / jacobian_cur * ( ( u11.cross( u2 ) ).transpose() * du1_Bmatrix + ( u1.cross( u11 ) ).transpose() * du2_Bmatrix ) +
        u3.dot( u11 ) / jacobian_cur * ( ( u2.cross( u3 ) ).transpose() * du1_Bmatrix + ( u3.cross( u1 ) ).transpose() * du2_Bmatrix );
    bilinear_form_trial.middleRows( 1, 1 ) =
        -u3.transpose() * du22_Bmatrix +
        1.0 / jacobian_cur * ( ( u22.cross( u2 ) ).transpose() * du1_Bmatrix + ( u1.cross( u22 ) ).transpose() * du2_Bmatrix ) +
        u3.dot( u22 ) / jacobian_cur * ( ( u2.cross( u3 ) ).transpose() * du1_Bmatrix + ( u3.cross( u1 ) ).transpose() * du2_Bmatrix );
    bilinear_form_trial.middleRows( 2, 1 ) =
        2 * ( -u3.transpose() * du12_Bmatrix +
              1.0 / jacobian_cur * ( ( u12.cross( u2 ) ).transpose() * du1_Bmatrix + ( u1.cross( u12 ) ).transpose() * du2_Bmatrix ) +
              u3.dot( u12 ) / jacobian_cur *
                  ( ( u2.cross( u3 ) ).transpose() * du1_Bmatrix + ( u3.cross( u1 ) ).transpose() * du2_Bmatrix ) );
    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );
    bilinear_form_test = _E * pow( _h, 3 ) / 12 / ( 1 - _nu * _nu ) * H * bilinear_form_trial;
    linear_form_test = bilinear_form_test;
}

template <typename T, NonlinearBendingStiffnessType nt>
template <NonlinearBendingStiffnessType tt>
typename std::enable_if<tt == NonlinearBendingStiffnessType::AllExcpet, void>::type NonlinearBendingStiffnessVisitor<T, nt>::Assemble(
    Matrix& bilinear_form_trial,
    Matrix& bilinear_form_test,
    Matrix& linear_form_value,
    Matrix& linear_form_test,
    const DomainShared_ptr domain,
    const Knot& u ) const
{
    const auto& current_config = domain->CurrentConfigGetter();
    const auto current_config_evals = current_config.EvalDerAllTensor( u, 2 );
    bilinear_form_trial.resize( 24, 3 * current_config_evals->size() );
    bilinear_form_trial.setZero();

    bilinear_form_test.resize( 24, 3 * current_config_evals->size() );
    bilinear_form_test.setZero();

    Eigen::Matrix<T, 3, 1> U1, U2, U3, U11, U12, U22, V1, V2, V3, u1, u2, u3, u11, u12, u22;

    U1 = domain->AffineMap( u, {1, 0} );
    U2 = domain->AffineMap( u, {0, 1} );
    U11 = domain->AffineMap( u, {2, 0} );
    U12 = domain->AffineMap( u, {1, 1} );
    U22 = domain->AffineMap( u, {0, 2} );
    U3 = U1.cross( U2 );
    const T jacobian_ref = U3.norm();
    const T inv_jacobian_ref = 1.0 / jacobian_ref;
    U3 *= inv_jacobian_ref;

    u1 = current_config.AffineMap( u, {1, 0} );
    u2 = current_config.AffineMap( u, {0, 1} );
    u11 = current_config.AffineMap( u, {2, 0} );
    u12 = current_config.AffineMap( u, {1, 1} );
    u22 = current_config.AffineMap( u, {0, 2} );
    u3 = ( u1.cross( u2 ) );
    const T jacobian_cur = u3.norm();
    const T inv_jacobian_cur = 1.0 / jacobian_cur;
    u3 *= inv_jacobian_cur;

    std::tie( V1, V2, V3 ) = Accessory::CovariantToContravariant( U1, U2, U3 );

    T V11, V12, V22;
    V11 = V1.dot( V1 );
    V22 = V2.dot( V2 );
    V12 = V1.dot( V2 );

    Matrix H( 3, 3 );
    H << V11 * V11, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12, V11 * V12, _nu * V11 * V22 + ( 1 - _nu ) * V12 * V12,
        V22 * V22, V22 * V12, V11 * V12, V22 * V12, .5 * ( ( 1 - _nu ) * V11 * V22 + ( 1 + _nu ) * V12 * V12 );

    Vector kappa( 3 );
    kappa( 0 ) = U11.dot( U3 ) - u11.dot( u3 );
    kappa( 1 ) = U22.dot( U3 ) - u22.dot( u3 );
    kappa( 2 ) = 2 * ( U12.dot( U3 ) - u12.dot( u3 ) );

    Vector m = _E * pow( _h, 3 ) / 12 / ( 1 - _nu * _nu ) * H * kappa;

    Matrix du1_Bmatrix, du2_Bmatrix, du11_Bmatrix, du22_Bmatrix, du12_Bmatrix;
    du1_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du2_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du11_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du22_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du12_Bmatrix.resize( 3, 3 * current_config_evals->size() );
    du1_Bmatrix.setZero();
    du2_Bmatrix.setZero();
    du11_Bmatrix.setZero();
    du22_Bmatrix.setZero();
    du12_Bmatrix.setZero();

    for ( int j = 0; j < current_config_evals->size(); ++j )
    {
        du1_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[1];
        du1_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[1];

        du2_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[2];
        du2_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[2];

        du11_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[3];
        du11_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[3];
        du11_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[3];

        du22_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[5];
        du22_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[5];
        du22_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[5];

        du12_Bmatrix( 0, 3 * j ) = ( *current_config_evals )[j].second[4];
        du12_Bmatrix( 1, 3 * j + 1 ) = ( *current_config_evals )[j].second[4];
        du12_Bmatrix( 2, 3 * j + 2 ) = ( *current_config_evals )[j].second[4];
    }

    const Vector A_alpha_beta_dot_m_vec = m( 0 ) * u11 + m( 1 ) * u22 + 2 * m( 2 ) * u12;

    const Matrix A_alpha_beta_dot_m_mat = Accessory::CrossProductMatrix( A_alpha_beta_dot_m_vec );

    const Matrix A1cu2minusA2cu1 =
        Accessory::CrossProductMatrix( u1 ) * du2_Bmatrix - Accessory::CrossProductMatrix( u2 ) * du1_Bmatrix;
    const T A_alpha_beta_dot_A3 = ( u3.transpose() * A_alpha_beta_dot_m_vec )( 0 );

    bilinear_form_test.middleRows( 0, 3 ) = -inv_jacobian_cur * A_alpha_beta_dot_m_mat * du1_Bmatrix;
    bilinear_form_trial.middleRows( 0, 3 ) = du2_Bmatrix;
    bilinear_form_test.middleRows( 3, 3 ) = bilinear_form_trial.middleRows( 0, 3 );
    bilinear_form_trial.middleRows( 3, 3 ) = bilinear_form_test.middleRows( 0, 3 );

    bilinear_form_test.middleRows( 6, 1 ) = std::pow( inv_jacobian_cur, 2 ) * A_alpha_beta_dot_m_vec.transpose() * A1cu2minusA2cu1;
    bilinear_form_trial.middleRows( 6, 1 ) = u3.transpose() * A1cu2minusA2cu1;
    bilinear_form_trial.middleRows( 7, 1 ) = bilinear_form_test.middleRows( 6, 1 );
    bilinear_form_test.middleRows( 7, 1 ) = bilinear_form_trial.middleRows( 6, 1 );

    bilinear_form_trial.middleRows( 8, 1 ) = u3.transpose() * A1cu2minusA2cu1;
    bilinear_form_test.middleRows( 8, 1 ) =
        -A_alpha_beta_dot_A3 * 3 * std::pow( inv_jacobian_cur, 2 ) * ( u3.transpose() * A1cu2minusA2cu1 );

    bilinear_form_trial.middleRows( 9, 3 ) = A_alpha_beta_dot_A3 * Accessory::CrossProductMatrix( u3 ) * du1_Bmatrix;
    bilinear_form_test.middleRows( 9, 3 ) = inv_jacobian_cur * du2_Bmatrix;

    bilinear_form_trial.middleRows( 12, 3 ) = bilinear_form_test.middleRows( 9, 3 );
    bilinear_form_test.middleRows( 12, 3 ) = bilinear_form_trial.middleRows( 9, 3 );

    bilinear_form_trial.middleRows( 15, 3 ) = A1cu2minusA2cu1;
    bilinear_form_test.middleRows( 15, 3 ) = A_alpha_beta_dot_A3 * std::pow( inv_jacobian_cur, 2 ) * A1cu2minusA2cu1;

    // check
    bilinear_form_trial.middleRows( 18, 3 ) =
        -inv_jacobian_cur * ( m( 0 ) * du11_Bmatrix + m( 1 ) * du22_Bmatrix + 2 * m( 2 ) * du12_Bmatrix );
    bilinear_form_test.middleRows( 18, 3 ) = A1cu2minusA2cu1 - u3 * ( u3.transpose() * A1cu2minusA2cu1 );

    bilinear_form_trial.middleRows( 21, 3 ) = bilinear_form_test.middleRows( 18, 3 );
    bilinear_form_test.middleRows( 21, 3 ) = bilinear_form_trial.middleRows( 18, 3 );

    linear_form_value.resize( 1, 1 );
    linear_form_value.setZero();
    linear_form_test.resize( 1, 3 * current_config_evals->size() );
    linear_form_test.setZero();
}