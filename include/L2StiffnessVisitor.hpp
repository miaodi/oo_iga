#pragma once

#include "StiffnessVisitor.hpp"

template <typename T>
class L2StiffnessVisitor : public StiffnessVisitor<2, 1, T>
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
    L2StiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<2, 1, T>( body_force )
    {
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const override
    {
        auto evals = domain->EvalDerAllTensor( u );
        linear_form_value.resize( 1, 1 );
        linear_form_value( 0, 0 ) = 0;
        linear_form_test.resize( 1, evals->size() );
        bilinear_form_trail.resize( 1, evals->size() );
        for ( int j = 0; j < evals->size(); ++j )
        {
            linear_form_test( 0, j ) = ( *evals )[j].second[0];
        }

        bilinear_form_trail = linear_form_test;
        bilinear_form_test = bilinear_form_trail;
    }
};

template <int d, int N, typename T>
class H2StiffnessVisitor : public StiffnessVisitor<N, 1, T, d>
{
public:
    using Knot = typename StiffnessVisitor<N, 1, T, d>::Knot;
    using Quadrature = typename StiffnessVisitor<N, 1, T, d>::Quadrature;
    using QuadList = typename StiffnessVisitor<N, 1, T, d>::QuadList;
    using KnotSpan = typename StiffnessVisitor<N, 1, T, d>::KnotSpan;
    using KnotSpanlist = typename StiffnessVisitor<N, 1, T, d>::KnotSpanlist;
    using LoadFunctor = typename StiffnessVisitor<N, 1, T, d>::LoadFunctor;
    using Matrix = typename StiffnessVisitor<N, 1, T, d>::Matrix;
    using Vector = typename StiffnessVisitor<N, 1, T, d>::Vector;
    using DomainShared_ptr = typename StiffnessVisitor<N, 1, T, d>::DomainShared_ptr;
    using DataType = T;

public:
    H2StiffnessVisitor( const LoadFunctor& body_force ) : StiffnessVisitor<N, 1, T, d>( body_force )
    {
    }

protected:
    virtual void IntegralElementAssembler( Matrix& bilinear_form_trail,
                                           Matrix& bilinear_form_test,
                                           Matrix& linear_form_value,
                                           Matrix& linear_form_test,
                                           const DomainShared_ptr domain,
                                           const Knot& u ) const override
    {
        auto evals = domain->Eval2PhyDerAllTensor( u );
        linear_form_value.resize( 6, 1 );
        linear_form_value( 0, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[0];
        linear_form_value( 1, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[1];
        linear_form_value( 2, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[2];
        linear_form_value( 3, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[3];
        linear_form_value( 4, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[4];
        linear_form_value( 5, 0 ) = this->_bodyForceFunctor( domain->AffineMap( u ) )[5];
        linear_form_test.resize( 6, evals->size() );
        bilinear_form_trail.resize( 6, evals->size() );
        for ( int j = 0; j < evals->size(); ++j )
        {
            linear_form_test( 0, j ) = ( *evals )[j].second[0];
            linear_form_test( 1, j ) = ( *evals )[j].second[1];
            linear_form_test( 2, j ) = ( *evals )[j].second[2];
            linear_form_test( 3, j ) = ( *evals )[j].second[3];
            linear_form_test( 4, j ) = ( *evals )[j].second[4];
            linear_form_test( 5, j ) = ( *evals )[j].second[5];
        }

        bilinear_form_trail = linear_form_test;
        bilinear_form_test = bilinear_form_trail;
    }
};
