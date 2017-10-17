//
// Created by di miao on 10/16/17.
//

#pragma once

#include "Visitor.hpp"
#include "DofMapper.hpp"
#include "Edge.hpp"
#include "Surface.hpp"
#include "Vertex.hpp"

template<int N, typename T>
class PoissonMapper : public Visitor<2, N, T>, Visitor<1, N, T>, Visitor<0, N, T>
{
public:
    PoissonMapper(DofMapper<N, T> &dofMap) : _dofMap(dofMap)
    {

    }

    void
    Visit(Element<1,N,T> *g)
    {
        auto edge = dynamic_cast<Edge<N,T>*>(g);
        if (!edge->GetMatchInfo())
        {
            if (edge->IsDirichlet())
            {
                auto tmp = edge->ExclusiveIndices(0);
                for (const auto &i:*tmp)
                {
                    _dofMap.DirichletDofInserter(edge->Parent(0).lock()->GetDomain(), i);
                }
            }
        }
        else
        {
            if (!edge->Slave())
                return;
            auto tmp = edge->ExclusiveIndices(0);
            for (const auto &i:*tmp)
            {
                _dofMap.SlaveDofInserter(edge->Parent(0).lock()->GetDomain(), i);
            }
        }

    }

    void
    Visit(Element<0,N,T> *g){
        auto vertex = dynamic_cast<Vertex<N,T>*>(g);

    }

    void
    Visit(Element<2,N,T> *g)
    {
        auto surface = dynamic_cast<Surface<N,T>*>(g);
        _dofMap.DomainLabel(surface->GetDomain());
        _dofMap.PatchDofSetter(surface->GetDomain(), surface->GetDomain()->GetDof());
        surface->EdgeAccept(*this);
    }

private:
    DofMapper<N, T> &_dofMap;
};