//
// Created by di miao on 10/9/17.
//

#pragma once

#include "Topology.hpp"
#include "Surface.hpp"
#include "Visitor.hpp"

template <int d, int N, typename T>
class Visitor;

template <int N, typename T>
class Vertex;

template <int N, typename T>
class Surface;

template <int N, typename T>
struct ComputeNormal;

template <int N, typename T>
class Edge : public Element<1, N, T>, public std::enable_shared_from_this<Edge<N, T>>
{
  public:
    typedef typename Element<1, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<1, N, T>::Coordinate Coordinate;
    typedef typename Element<1, N, T>::CoordinatePairList CoordinatePairList;
    using PhyPts = typename PhyTensorBsplineBasis<2, N, T>::PhyPts;
    using EdgeShared_Ptr = typename std::shared_ptr<Edge<N, T>>;

    Edge(const Orientation &orient = Orientation::west)
        : Element<1, N, T>(), _position(orient){};

    Edge(DomainShared_ptr m, const Orientation &orient) : Element<1, N, T>(m), _position(orient) {}

    std::unique_ptr<std::vector<int>>
    Indices(const int &layer) const
    {
        std::unique_ptr<std::vector<int>> res(new std::vector<int>);
        auto parent = _parents[0].lock();
        auto domain = parent->GetDomain();
        switch (_position)
        {
        case Orientation::west:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(0, i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case Orientation::east:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(0, domain->GetDof(0) - 1 - i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case Orientation::north:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(1, domain->GetDof(1) - 1 - i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case Orientation::south:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(1, i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        }
        std::sort(res->begin(), res->end());
        return res;
    };

    void
    PrintInfo() const
    {
        switch (_position)
        {
        case Orientation::west:
        {
            std::cout << "West edge:" << std::endl;
            break;
        }
        case Orientation::east:
        {
            std::cout << "East edge:" << std::endl;
            break;
        }
        case Orientation::north:
        {
            std::cout << "North edge:" << std::endl;
            break;
        }
        case Orientation::south:
        {
            std::cout << "South edge:" << std::endl;
            break;
        }
        }
    }

    //! Return the element coordinates in parametric domain. (Each element in the vector is composed with two points,
    //! i.e. Southeast and Northwest.)
    void
    KnotSpansGetter(CoordinatePairList &knotspanslist)
    {
        switch (_position)
        {
        case Orientation::west:
        {
            auto knot_y = this->_domain->KnotVectorGetter(1);
            auto knot_x = this->_domain->DomainStart(0);
            auto knotspan_y = knot_y.KnotSpans();
            knotspanslist.reserve(knotspan_y.size());
            for (const auto &i : knotspan_y)
            {
                Coordinate _begin;
                _begin << knot_x, i.first;
                Coordinate _end;
                _end << knot_x, i.second;
                knotspanslist.push_back({_begin, _end});
            }
            break;
        }
        case Orientation::east:
        {
            auto knot_y = this->_domain->KnotVectorGetter(1);
            auto knot_x = this->_domain->DomainEnd(0);
            auto knotspan_y = knot_y.KnotSpans();
            knotspanslist.reserve(knotspan_y.size());
            for (const auto &i : knotspan_y)
            {
                Coordinate _begin;
                _begin << knot_x, i.first;
                Coordinate _end;
                _end << knot_x, i.second;
                knotspanslist.push_back({_begin, _end});
            }
            break;
        }
        case Orientation::south:
        {
            auto knot_y = this->_domain->DomainStart(1);
            auto knot_x = this->_domain->KnotVectorGetter(0);
            auto knotspan_x = knot_x.KnotSpans();
            knotspanslist.reserve(knotspan_x.size());
            for (const auto &i : knotspan_x)
            {
                Coordinate _begin;
                _begin << i.first, knot_y;
                Coordinate _end;
                _end << i.second, knot_y;
                knotspanslist.push_back({_begin, _end});
            }
            break;
        }
        case Orientation::north:
        {
            auto knot_y = this->_domain->DomainEnd(1);
            auto knot_x = this->_domain->KnotVectorGetter(0);
            auto knotspan_x = knot_x.KnotSpans();
            knotspanslist.reserve(knotspan_x.size());
            for (const auto &i : knotspan_x)
            {
                Coordinate _begin;
                _begin << i.first, knot_y;
                Coordinate _end;
                _end << i.second, knot_y;
                knotspanslist.push_back({_begin, _end});
            }
            break;
        }
        }
    }

    T Measure() const
    {
        return 0;
    }

    void
        Accept(Visitor<1, N, T> &visitor)
    {
        visitor.Visit(this);
    };

    void
    PrintOrient() const
    {
        std::cout << _position << std::endl;
    }

    Orientation
    GetOrient() const
    {
        return _position;
    }

    void
    ParentSetter(const std::shared_ptr<Surface<N, T>> &parent)
    {
        _parents.push_back(std::weak_ptr<Surface<N, T>>(parent));
    }

    auto
    Parent(const int &i) const
    {
        return _parents[i];
    }

    void
    PrintIndices(const int &layerNum) const
    {
        std::cout << "Activated Dofs on this edge are: ";
        Element<1, N, T>::PrintIndices(layerNum);
    }

    PhyPts NormalDirection(const Coordinate &u) const
    {
        ComputeNormal<N, T> temp;
        return temp.compute(this, u);
    }

  protected:
    Orientation _position;
    std::vector<std::weak_ptr<Surface<N, T>>> _parents;
};

template <int N, typename T>
struct ComputeNormal
{
    using Pts = typename PhyTensorBsplineBasis<1, N, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<1, N, T>::PhyPts;

    virtual PhyPts compute(const Edge<N, T> *edge_ptr, const Pts &u) = 0;
};

template <typename T>
struct ComputeNormal<2, T>
{
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<1, 2, T>::PhyPts;

    virtual PhyPts compute(const Edge<2, T> *edge_ptr, const Pts &u)
    {
        PhyPts normal;
        switch (edge_ptr->GetOrient())
        {
        case Orientation::west:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << -tangent(1), tangent(0);
            normal.normalize();
            break;
        }
        case Orientation::east:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << tangent(1), -tangent(0);
            normal.normalize();
            break;
        }
        case Orientation::north:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << -tangent(1), tangent(0);
            normal.normalize();
            break;
        }
        case Orientation::south:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << tangent(1), -tangent(0);
            normal.normalize();
            break;
        }
        }
        return normal;
    }
};
