//
// Created by di miao on 10/9/17.
//

#ifndef OO_IGA_SURFACE_H
#define OO_IGA_SURFACE_H

#include "Topology.hpp"
#include "Edge.hpp"
#include "Visitor.hpp"

template <int d, int N, typename T>
class Visitor;

template <int N, typename T>
class Edge;

template <int N, typename T>
class Vertex;

template <int N, typename T>
class Surface : public Element<2, N, T>, public std::enable_shared_from_this<Surface<N, T>>
{
  public:
    typedef typename Element<2, N, T>::PhyPts PhyPts;
    typedef typename Element<2, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<2, N, T>::Coordinate Coordinate;
    typedef typename Element<2, N, T>::CoordinatePairList CoordinatePairList;
    using SurfaceShared_Ptr = typename std::shared_ptr<Surface<N, T>>;

    Surface() : Element<2, N, T>(){};

    Surface(DomainShared_ptr m) : Element<2, N, T>(m)
    {
    }

    // shared_from_this requires that there be at least one shared_ptr instance that owns *this
    void SurfaceInitialize()
    {
        _edges[0] = std::make_shared<Edge<N, T>>(MakeEdge(Orientation::south), Orientation::south);
        _edges[0]->ParentSetter(this->shared_from_this());

        _edges[1] = std::make_shared<Edge<N, T>>(MakeEdge(Orientation::east), Orientation::east);
        _edges[1]->ParentSetter(this->shared_from_this());

        _edges[2] = std::make_shared<Edge<N, T>>(MakeEdge(Orientation::north), Orientation::north);
        _edges[2]->ParentSetter(this->shared_from_this());

        _edges[3] = std::make_shared<Edge<N, T>>(MakeEdge(Orientation::west), Orientation::west);
        _edges[3]->ParentSetter(this->shared_from_this());
    }

    virtual std::unique_ptr<std::vector<int>> Indices(const int &layer) const
    {
        return this->_domain->Indices();
    }

    void Accept(Visitor<2, N, T> &a)
    {
        a.Visit(this);
    };

    void EdgeAccept(Visitor<1, N, T> &a)
    {
        for (auto &i : _edges)
        {
            i->Accept(a);
        }
    };

    auto EdgePointerGetter(const int &i)
    {
        return _edges[i];
    }

    void PrintIndices(const int &layerNum = 0) const
    {
        std::cout << "Activated Dofs on this surface are: ";
        Element<2, N, T>::PrintIndices(layerNum);
    }

    //! Return the element coordinates in parametric domain. (Each element in the vector is composed with two points,
    //! i.e. Southeast and Northwest.)
    void KnotSpansGetter(CoordinatePairList &knotspanslist)
    {
        auto knotspan_x = this->_domain->KnotVectorGetter(0).KnotSpans();
        auto knotspan_y = this->_domain->KnotVectorGetter(1).KnotSpans();
        knotspanslist.reserve(knotspan_x.size() * knotspan_y.size());
        for (const auto &i : knotspan_x)
        {
            for (const auto &j : knotspan_y)
            {
                Coordinate _begin;
                _begin << i.first, j.first;
                Coordinate _end;
                _end << i.second, j.second;
                knotspanslist.push_back({_begin, _end});
            }
        }
    };

    //TODO: Finish the calculation of area;
    T Measure() const
    {
        return 0;
    }

    void PrintEdgeInfo() const
    {
        for (const auto &i : _edges)
        {
            i->PrintInfo();
        }
        std::cout << std::endl;
    }

    T Jacobian(const Coordinate &u) const
    {
        return this->_domain->Jacobian(u);
    }

  protected:
    std::array<std::shared_ptr<Edge<N, T>>, 4> _edges;

    std::shared_ptr<PhyTensorBsplineBasis<1, N, T>> MakeEdge(const Orientation &orient) const
    {
        switch (orient)
        {
        case Orientation::west:
        {
            return this->_domain->MakeHyperPlane(0, 0);
        }
        case Orientation::east:
        {
            return this->_domain->MakeHyperPlane(0, this->_domain->GetDof(0) - 1);
        }
        case Orientation::south:
        {
            return this->_domain->MakeHyperPlane(1, 0);
        }
        case Orientation::north:
        {
            return this->_domain->MakeHyperPlane(1, this->_domain->GetDof(1) - 1);
        }
        }
    }
};

#endif //OO_IGA_SURFACE_H
