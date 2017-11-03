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

    Edge(const Orientation &orient = west)
        : Element<1, N, T>(), _position(orient), _matched(false){};

    Edge(DomainShared_ptr m, const Orientation &orient, std::shared_ptr<Vertex<N, T>> &begin,
         std::shared_ptr<Vertex<N, T>> &end, const bool &dirichlet)
        : Element<1, N, T>(m), _position(orient), _matched(false), _Dirichlet(dirichlet)
    {
        _vertices[0] = begin;
        _vertices[1] = end;
    };

    std::unique_ptr<std::vector<int>>
    Indices(const int &layer) const
    {
        std::unique_ptr<std::vector<int>> res(new std::vector<int>);
        auto parent = _parents[0].lock();
        auto domain = parent->GetDomain();
        switch (_position)
        {
        case west:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(0, i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case east:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(0, domain->GetDof(0) - 1 - i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case north:
        {
            for (int i = 0; i <= layer; ++i)
            {
                auto tmp = domain->HyperPlaneIndices(1, domain->GetDof(1) - 1 - i);
                res->insert(res->end(), tmp->begin(), tmp->end());
            }
            break;
        }
        case south:
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

    std::unique_ptr<std::vector<int>>
    ExclusiveIndices(const int &layer) const
    {
        auto res = this->Indices(layer);
        std::unique_ptr<std::vector<int>> temp;
        for (int i = 0; i < _vertices.size(); ++i)
        {
            temp = _vertices[i]->Indices(layer);
            std::vector<int> diff;
            std::set_difference(res->begin(), res->end(), temp->begin(), temp->end(), std::back_inserter(diff));
            *res = diff;
        }
        return res;
    }

    bool IsDirichlet() const
    {
        return _Dirichlet;
    }

    void
    PrintInfo() const
    {
        switch (_position)
        {
        case west:
        {
            std::cout << "West edge:" << std::endl;
            break;
        }
        case east:
        {
            std::cout << "East edge:" << std::endl;
            break;
        }
        case north:
        {
            std::cout << "North edge:" << std::endl;
            break;
        }
        case south:
        {
            std::cout << "South edge:" << std::endl;
            break;
        }
        }
        auto begin = _vertices[0]->GetDomain();
        auto end = _vertices[1]->GetDomain();
        PhyPts beginPts = begin->Position();
        PhyPts endPts = end->Position();
        std::cout << "Starting Point: "
                  << "(";
        for (int i = 0; i < N - 1; i++)
        {
            std::cout << beginPts(i) << ", ";
        }
        std::cout << beginPts(N - 1) << ")" << std::endl;
        std::cout << "Ending Point: "
                  << "(";
        for (int i = 0; i < N - 1; i++)
        {
            std::cout << endPts(i) << ", ";
        }
        std::cout << endPts(N - 1) << ")" << std::endl;
        if (IsDirichlet())
        {
            std::cout << "Dirichlet boundary." << std::endl;
        }
        else
        {
            std::cout << "Neumann boundary." << std::endl;
        }
        std::cout << std::endl;
        ;
    }

    //! Return the element coordinates in parametric domain. (Each element in the vector is composed with two points,
    //! i.e. Southeast and Northwest.)
    void
    KnotSpansGetter(CoordinatePairList &knotspanslist)
    {
        switch (_position)
        {
        case west:
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
        case east:
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
        case south:
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
        case north:
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

    PhyPts
    GetStartCoordinate() const
    {
        return _vertices[0]->GetDomain()->Position();
    }

    PhyPts
    GetEndCoordinate() const
    {
        return _vertices[1]->GetDomain()->Position();
    }

    bool
    IsMatched() const
    {
        return _matched;
    }

    bool
    IsSlave() const
    {
        return _slave;
    }

    auto
    Counterpart() const
    {
        return _pair;
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

    void
    PrintExclusiveIndices(const int &layerNum) const
    {
        std::cout << "Activated exclusive Dofs on this edge are: ";
        Element<1, N, T>::PrintExclusiveIndices(layerNum);
    }

    //
    bool
    Match(std::shared_ptr<Edge<N, T>> &counterpart)
    {

        if (_matched == true || counterpart->_matched == true)
        {
            return true;
        }
        if (((GetStartCoordinate() == counterpart->GetStartCoordinate()) &&
             (GetEndCoordinate() == counterpart->GetEndCoordinate())) ||
            ((GetStartCoordinate() == counterpart->GetEndCoordinate()) &&
             (GetEndCoordinate() == counterpart->GetStartCoordinate())))
        {
            _pair = counterpart;
            _matched = true;
            counterpart->_pair = this->shared_from_this();
            counterpart->_matched = true;
            if (this->GetDomain()->GetDof(0) > counterpart->GetDomain()->GetDof(0))
            {
                _slave = true;
            }
            else
            {
                counterpart->_slave = true;
            }
            return true;
        }
        return false;
    }

    PhyPts NormalDirection(const Coordinate &u) const
    {
        ComputeNormal<N, T> temp;
        return temp.compute(this, u);
    }

  protected:
    Orientation _position;
    bool _matched;
    bool _slave{false};
    std::array<std::shared_ptr<Vertex<N, T>>, 2> _vertices;
    std::weak_ptr<Edge<N, T>> _pair;
    bool _Dirichlet{false};
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
        case west:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << -tangent(1), tangent(0);
            normal.normalize();
            break;
        }
        case east:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << tangent(1), -tangent(0);
            normal.normalize();
            break;
        }
        case north:
        {
            PhyPts tangent = edge_ptr->GetDomain()->AffineMap(u, {1});
            normal << -tangent(1), tangent(0);
            normal.normalize();
            break;
        }
        case south:
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
