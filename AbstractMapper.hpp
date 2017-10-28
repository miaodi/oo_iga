//
// Created by di miao on 10/23/17.
//

#pragma once

#include "Visitor.hpp"
#include "DofMapper.hpp"
#include "Edge.hpp"
#include "Surface.hpp"
#include "Vertex.hpp"

template <int layer, int N, typename T>
class AbstractMapper : public Visitor<2, N, T>, Visitor<1, N, T>, Visitor<0, N, T>
{
  public:
    AbstractMapper(DofMapper<N, T> &dofMap) : _dofMap(dofMap), _numOfLayer(layer)
    {
    }

    //    Visit edges to collect d.o.f information
    void
        Visit(Element<1, N, T> *g)
    {
        auto edge = dynamic_cast<Edge<N, T> *>(g);
        _dofMap.EdgeIndicesInserter(edge, *edge->ExclusiveIndices(_numOfLayer));
        //        If not matched to other edges, see if it is a Dirichlet boundary
        if (!edge->IsMatched())
        {
            if (edge->IsDirichlet())
            {
                auto tmp = edge->ExclusiveIndices(_numOfLayer);
                for (const auto &i : *tmp)
                {
                    _dofMap.DirichletDofInserter(edge->Parent(0).lock()->GetDomain(), i);
                }
            }
        }
        else
        {

            //          See if it is a slave edge.  I
            if (!edge->IsSlave())
                return;
            auto tmp = edge->ExclusiveIndices(_numOfLayer);
            for (const auto &i : *tmp)
            {
                _dofMap.SlaveDofInserter(edge->Parent(0).lock()->GetDomain(), i);
            }
        }
    }

    //    Visit vertices to collect d.o.f information
    void
        Visit(Element<0, N, T> *g)
    {
        auto vertex = dynamic_cast<Vertex<N, T> *>(g);

        //        If it is not Dirichlet and is slave push d.o.f associated with this vertex to slave d.o.f
        if (!vertex->IsDirichlet() && vertex->IsSlave())
        {
            auto tmp = vertex->ExclusiveIndices(_numOfLayer);
            for (const auto &i : *tmp)
            {
                _dofMap.SlaveDofInserter(vertex->Parent(0).lock()->Parent(0).lock()->GetDomain(), i);
            }
        }

        //            If It is Dirichlet push d.o.f associated with this vertex to Dirichlet.
        else if (vertex->IsDirichlet())
        {
            auto tmp = vertex->ExclusiveIndices(_numOfLayer);
            for (const auto &i : *tmp)
            {
                _dofMap.DirichletDofInserter(vertex->Parent(0).lock()->Parent(0).lock()->GetDomain(), i);
            }
        }
    }

    // Visit surfaces to collect d.o.f information
    void
        Visit(Element<2, N, T> *g)
    {
        auto surface = dynamic_cast<Surface<N, T> *>(g);
        _dofMap.DomainLabel(surface->GetDomain());
        _dofMap.PatchDofSetter(surface->GetDomain(), surface->GetDomain()->GetDof());
        surface->EdgeAccept(*this);
        surface->VertexAccept(*this);
    }

    virtual ~AbstractMapper() = 0;

  protected:
    DofMapper<N, T> &_dofMap;
    int _numOfLayer;
};

template <int layer, int N, typename T>
AbstractMapper<layer,N,T>::~AbstractMapper(){}