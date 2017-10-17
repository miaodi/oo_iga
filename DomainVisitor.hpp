//
// Created by di miao on 10/17/17.
//

#pragma once

#include "Visitor.hpp"
#include "Topology.hpp"
#include "QuadratureRule.h"
#include "DofMapper.hpp"

template<int d, int N, typename T>
class DomainVisitor : public Visitor<d, N, T>
{
public:
    using Coordinate = typename QuadratureRule<T>::Coordinate;
    using Quadrature = typename QuadratureRule<T>::Quadrature;
    using CoordinatePair = std::pair<Coordinate, Coordinate>;
    using

    DomainVisitor() {};

    void
    Visit(Element<d, N, T> *g)
    {

    }

//    Initialize quadrature rule
    void
    Initialize(Element<d, N, T> *g, QuadratureRule<T> &quad_rule,)
    {
        if (d == 0)
        {
            quad_rule.SetUpQuadrature(1);
        }
        else
        {
            auto domain = g->GetDomain();
            int max_degree = 0;
            for (int i = 0; i < d; i++)
            {
                max_degree = std::max(max_degree, domain->GetDegree(i));
            }
            quad_rule.SetUpQuadrature(max_degree + 1);
        }
    }

protected:
    DofMapper<N,T> _dofMapper;

};
