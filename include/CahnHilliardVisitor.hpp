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
        bilinear_form_trail = D * laplacian_c * ( 1 - 2 * c ) * dc + c * D * ( 1 - c ) * bilinear_form_test;
        bilinear_form_test = bilinear_form_trail;
        linear_form_test = bilinear_form_test;
        linear_form_value( 0, 0 ) = c * D * ( 1 - c ) * laplacian_c;
    }

protected:
    T* _disp{nullptr};
    T D;
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

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const
    {
        auto evals = domain->Eval1PhyDerAllTensor( u );
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
        bilinear_form_trail = D * laplacian_c * ( 1 - 2 * c ) * dc + c * D * ( 1 - c ) * bilinear_form_test;
        bilinear_form_test = bilinear_form_trail;
        linear_form_test = bilinear_form_test;
        linear_form_value( 0, 0 ) = c * D * ( 1 - c ) * laplacian_c;
    }

protected:
    T* _vel{nullptr};
    T D;
};