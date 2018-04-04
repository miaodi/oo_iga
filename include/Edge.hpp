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

    Edge(DomainShared_ptr m, const Orientation &orient, std::shared_ptr<Vertex<N, T>> &begin,
         std::shared_ptr<Vertex<N, T>> &end)
        : Element<1, N, T>(m), _position(orient), _matched(false)
    {
        _vertices[0] = begin;
        _vertices[1] = end;
    };

    virtual std::vector<int> Indices(const int &dimension, const int &layer) const
    {
        std::vector<int> res;
        auto parent = _parents[0].lock();
        auto domain = parent->GetDomain();
        std::unique_ptr<std::vector<int>> indices;
        switch (_position)
        {
        case Orientation::west:
        {
            for (int i = 0; i <= layer; ++i)
            {
                indices = domain->HyperPlaneIndices(0, i);
                for (auto &i : *indices)
                {
                    for (int j = 0; j < dimension; j++)
                    {
                        res.push_back(dimension * i + j);
                    }
                }
            }
            break;
        }
        case Orientation::east:
        {
            for (int i = 0; i <= layer; ++i)
            {
                indices = domain->HyperPlaneIndices(0, domain->GetDof(0) - 1 - i);
                for (auto &i : *indices)
                {
                    for (int j = 0; j < dimension; j++)
                    {
                        res.push_back(dimension * i + j);
                    }
                }
            }
            break;
        }
        case Orientation::north:
        {
            for (int i = 0; i <= layer; ++i)
            {
                indices = domain->HyperPlaneIndices(1, domain->GetDof(1) - 1 - i);
                for (auto &i : *indices)
                {
                    for (int j = 0; j < dimension; j++)
                    {
                        res.push_back(dimension * i + j);
                    }
                }
            }
            break;
        }
        case Orientation::south:
        {
            for (int i = 0; i <= layer; ++i)
            {
                indices = domain->HyperPlaneIndices(1, i);
                for (auto &i : *indices)
                {
                    for (int j = 0; j < dimension; j++)
                    {
                        res.push_back(dimension * i + j);
                    }
                }
            }
            break;
        }
        }

        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> ExclusiveIndices(const int &dimension, const int &layer) const
    {
        auto res = this->Indices(dimension, layer);
        std::vector<int> temp;
        for (int i = 0; i < _vertices.size(); ++i)
        {
            temp = _vertices[i]->Indices(dimension, layer);
            std::vector<int> diff;
            std::set_difference(res.begin(), res.end(), temp.begin(), temp.end(), std::back_inserter(diff));
            res = diff;
        }
        return res;
    }

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

        PhyPts beginPts = _vertices[0]->Position();
        PhyPts endPts = _vertices[1]->Position();

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

        if (IsSlave())
        {
            std::cout << "Slave edge." << std::endl;
        }
        else
        {
            std::cout << "Master edge." << std::endl;
        }
        if (IsMatched())
        {
            std::cout << "Matched." << std::endl;
        }
        else
        {
            std::cout << "Not matched." << std::endl;
        }

        std::cout << std::endl;
    }

    bool
    Match(std::shared_ptr<Edge<N, T>> counterpart)
    {
        T tol = std::numeric_limits<T>::epsilon() * 1e3;
        if (_matched == true || counterpart->_matched == true)
        {
            return true;
        }
        if (((GetStartCoordinate() - counterpart->GetStartCoordinate()).norm() < tol && (GetEndCoordinate() - counterpart->GetEndCoordinate()).norm() < tol) ||
            ((GetStartCoordinate() - counterpart->GetEndCoordinate()).norm() < tol && (GetEndCoordinate() - counterpart->GetStartCoordinate()).norm() < tol))
        {
            _pair = counterpart;
            _matched = true;
            counterpart->_pair = this->shared_from_this();
            counterpart->_matched = true;
            if (this->GetDomain()->GetDof(0) > counterpart->GetDomain()->GetDof(0))
            {
                _slave = true;
                counterpart->_slave = false;
            }
            else
            {
                _slave = false;
                counterpart->_slave = true;
            }
            return true;
        }
        return false;
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

    PhyPts
    GetStartCoordinate() const
    {
        return _vertices[0]->Position();
    }

    PhyPts
    GetEndCoordinate() const
    {
        return _vertices[1]->Position();
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
    PrintIndices(const int &dimension, const int &layerNum) const
    {
        std::cout << "Activated Dofs of " << dimension << " dimensional field on this edge are: ";
        Element<1, N, T>::PrintIndices(dimension, layerNum);
    }

    void
    PrintExclusiveIndices(const int &dimension, const int &layerNum) const
    {
        std::cout << "Activated Dofs of " << dimension << " dimensional field on this edge are: ";
        Element<1, N, T>::PrintExclusiveIndices(dimension, layerNum);
    }

    PhyPts NormalDirection(const Coordinate &u) const
    {
        ComputeNormal<N, T> temp;
        return temp.compute(this, u);
    }

    auto VertexPointerGetter(const int &i)
    {
        ASSERT(i >= 0 && i <= 1, "Invalid vertex index.\n");
        return _vertices[i];
    }

  protected:
    Orientation _position;
    bool _matched;
    bool _slave{false};
    std::array<std::shared_ptr<Vertex<N, T>>, 2> _vertices;
    std::weak_ptr<Edge<N, T>> _pair;
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
