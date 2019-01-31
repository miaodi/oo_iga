#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class CH4thStiffnessVisitor : public StiffnessVisitor<2, 1, T>
{
public:
    using Knot = typename StiffnessVisitor<2, 1, T>::Knot;
    using Quadrature = typename StiffnessVisitor<2, 1, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<2, 1, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<2, 1, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<2, 1, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<2, 1, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<2, 1, T>::Matrix;
    using Vector = typename StiffnessVisitor<2, 1, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<2, 1, T>::DomainShared_ptr;
    using DataType = T;
    static const int Dim = StiffnessVisitor<2, 1, T>::Dim;

public:
    CH4thStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<2, 1, T>( body_force )
    {
    }

    void SetStateData( T* disp )
    {
        _disp = disp;
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const
    {
        auto evals = domain->Eval2PhyDerAllTensor( u );
        linear_form_value.resize( 1, 1 );
        T c = 0, laplacian_c = 0;
        Matrix dc( 1, evals->size() );
        bilinear_form_test.resize( 1, evals->size() );
        for ( int j = 0; j < evals->size(); ++j )
        {
            dc( 0, j ) = ( *evals )[j].second[0];
            c += dc( 0, j ) * *( _disp + ( *evals )[j].first );
            bilinear_form_test( 0, j ) = ( *evals )[j].second[3] + ( *evals )[j].second[5];
            laplacian_c += bilinear_form_test( 0, j ) * *( _disp + ( *evals )[j].first );
        }
        bilinear_form_trail = laplacian_c * ( 1 - 2 * c ) * dc + c * ( 1 - c ) * bilinear_form_test;
        bilinear_form_test = bilinear_form_trail;
        linear_form_test = bilinear_form_test;
        linear_form_value( 0, 0 ) = c * ( 1 - c ) * laplacian_c;
    }

protected:
    T* _disp{nullptr};
};

template <typename T>
class CH2ndStiffnessVisitor : public StiffnessVisitor<2, 1, T>
{
public:
    using Knot = typename StiffnessVisitor<2, 1, T>::Knot;
    using Quadrature = typename StiffnessVisitor<2, 1, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<2, 1, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<2, 1, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<2, 1, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<2, 1, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<2, 1, T>::Matrix;
    using Vector = typename StiffnessVisitor<2, 1, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<2, 1, T>::DomainShared_ptr;
    using DataType = T;
    static const int Dim = StiffnessVisitor<2, 1, T>::Dim;

public:
    CH2ndStiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<2, 1, T>( body_force )
    {
    }

    void SetStateData( T* disp )
    {
        _disp = disp;
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const
    {
        auto evals = domain->Eval2PhyDerAllTensor( u );
        linear_form_value.resize( 2, 1 );
        T c = 0, laplacian_c = 0;
        Matrix grad( 2, 1 );
        grad.setZero();
        Matrix dc( 1, evals->size() );
        Matrix lc( 1, evals->size() );
        bilinear_form_test.resize( 2, evals->size() );
        for ( int j = 0; j < evals->size(); ++j )
        {
            dc( 0, j ) = ( *evals )[j].second[0];
            c += dc( 0, j ) * *( _disp + ( *evals )[j].first );
            grad( 0, 0 ) += ( *evals )[j].second[1] * *( _disp + ( *evals )[j].first );
            grad( 1, 0 ) += ( *evals )[j].second[2] * *( _disp + ( *evals )[j].first );
            bilinear_form_test( 0, j ) = ( *evals )[j].second[1];
            bilinear_form_test( 1, j ) = ( *evals )[j].second[2];
            lc( 0, j ) = ( *evals )[j].second[3] + ( *evals )[j].second[5];
            laplacian_c += lc( 0, j ) * *( _disp + ( *evals )[j].first );
        }
        linear_form_value = ( 3000 * ( 1 - 6 * c + 6 * c * c ) + laplacian_c * ( 1 - 2 * c ) ) * grad;
        linear_form_test = bilinear_form_test;
        bilinear_form_trail = 18000 * ( 2 * c - 1 ) * grad * dc +
                              3000 * ( 1 - 6 * c + 6 * c * c ) * bilinear_form_test + ( 1 - 2 * c ) * grad * lc -
                              2 * laplacian_c * grad * dc + laplacian_c * ( 1 - 2 * c ) * bilinear_form_test;
    }

protected:
    T* _disp{nullptr};
};

template <typename T>
class CHMassVisitor : public StiffnessVisitor<2, 1, T>
{
public:
    using Knot = typename StiffnessVisitor<2, 1, T>::Knot;
    using Quadrature = typename StiffnessVisitor<2, 1, T>::Quadrature;
    using QuadList = typename StiffnessVisitor<2, 1, T>::QuadList;
    using KnotSpan = typename StiffnessVisitor<2, 1, T>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<2, 1, T>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<2, 1, T>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<2, 1, T>::Matrix;
    using Vector = typename StiffnessVisitor<2, 1, T>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<2, 1, T>::DomainShared_ptr;
    using DataType = T;
    static const int Dim = StiffnessVisitor<2, 1, T>::Dim;

public:
    CHMassVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<2, 1, T>( body_force )
    {
    }

    void SetStateData( T* vel )
    {
        _vel = vel;
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const
    {
        auto evals = domain->EvalDerAllTensor( u );
        T c_dot = 0;
        linear_form_test.resize( 1, evals->size() );
        linear_form_value.resize( 1, 1 );
        for ( int j = 0; j < evals->size(); ++j )
        {
            linear_form_test( 0, j ) = ( *evals )[j].second[0];
            c_dot += linear_form_test( 0, j ) * *( _vel + ( *evals )[j].first );
        }
        bilinear_form_trail = linear_form_test;
        bilinear_form_test = linear_form_test;
        linear_form_value( 0, 0 ) = c_dot;
    }

protected:
    T* _vel{nullptr};
};