//
// Created by di miao on 10/9/17.
//

#ifndef OO_IGA_SURFACE_H
#define OO_IGA_SURFACE_H

#include "Topology.hpp"
#include "Edge.hpp"
#include "Visitor.hpp"

template<int d, int N, typename T>
class Visitor;

template<int N, typename T>
class Edge;

template<int N, typename T>
class Vertex;

template<int N, typename T>
class Surface : public Element<2, N, T> {
public:
    typedef typename Element<2, N, T>::PhyPts PhyPts;
    typedef typename Element<2, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<2, N, T>::Coordinate Coordinate;
    typedef typename Element<2, N, T>::CoordinatePairList CoordinatePairList;

    Surface() : Element<2, N, T>() {};

    Surface(DomainShared_ptr m, const std::array<bool, 4> &boundary) : Element<2, N, T>(m), _Dirichlet(boundary) {
        for (int i = 0; i < 4; i++) {
            _vertices[i] = std::make_shared<Vertex<N, T>>(MakeVertex(i),
                                                          _Dirichlet[(i - 1 >= 0) ? (i - 1) : 3] || _Dirichlet[i]);
        }
        _edges[0] = std::make_shared<Edge<N, T>>(MakeEdge(south), south, _vertices[0], _vertices[1]);
        _edges[1] = std::make_shared<Edge<N, T>>(MakeEdge(east), east, _vertices[1], _vertices[2]);
        _edges[2] = std::make_shared<Edge<N, T>>(MakeEdge(north), north, _vertices[2], _vertices[3]);
        _edges[3] = std::make_shared<Edge<N, T>>(MakeEdge(west), west, _vertices[3], _vertices[0]);
    }


    void Accept(Visitor<2, N, T> &a) {
        a.Visit(this);
    };

    void EdgeAccept(Visitor<1, N, T> &a) {
        for (auto &i:_edges) {
            i->Accept(a);
        }
    };

    auto EdgePointerGetter(const int &i) {
        return _edges[i];
    }

    //! Return the element coordinates in parametric domain. (Each element in the vector is composed with two points,
    //! i.e. Southeast and Northwest.)
    void KnotSpansGetter(CoordinatePairList &knotspanslist) {
        auto knotspan_x = this->_domain->KnotVectorGetter(0).KnotSpans();
        auto knotspan_y = this->_domain->KnotVectorGetter(1).KnotSpans();
        knotspanslist.reserve(knotspan_x.size() * knotspan_y.size());
        for (const auto &i:knotspan_x) {
            for (const auto &j:knotspan_y) {
                Coordinate _begin;
                _begin << i.first, j.first;
                Coordinate _end;
                _end << i.second, j.second;
                knotspanslist.push_back({_begin, _end});
            }
        }
    };

//TODO: Finish the calculation of area;
    T Measure() const {
        return 0;
    }

    void Match(std::shared_ptr<Surface<N, T>> &counterpart) {
        for (auto &i:_edges) {
            for (auto &j:counterpart->_edges) {
                i->Match(j);
            }
        }
        for (auto &i:_vertices) {
            for (auto &j:counterpart->_vertices) {
                i->Match(j);
            }
        }
    }

    void PrintEdgeInfo() const {
        for (const auto &i:_edges) {
            i->PrintInfo();
        }
        std::cout << std::endl;
    }

    void PrintVertexInfo() const {
        for (const auto &i:_vertices) {
            i->PrintInfo();
        }
        std::cout << std::endl;
    }

    T Jacobian(const Coordinate &u) const {
        return this->_domain->Jacobian(u);
    }

protected:
    std::array<std::shared_ptr<Edge<N, T>>, 4> _edges;
    std::array<std::shared_ptr<Vertex<N, T>>, 4> _vertices;
    std::array<bool, 4> _Dirichlet;


    std::shared_ptr<PhyTensorBsplineBasis<1, N, T>> MakeEdge(const Orientation &orient) const {
        switch (orient) {
            case west: {
                return this->_domain->MakeHyperPlane(0, 0);
            }
            case east: {
                return this->_domain->MakeHyperPlane(0, this->_domain->GetDof(0) - 1);
            }
            case south: {
                return this->_domain->MakeHyperPlane(1, 0);
            }
            case north: {
                return this->_domain->MakeHyperPlane(1, this->_domain->GetDof(1) - 1);
            }
        }
    }

    std::shared_ptr<PhyTensorBsplineBasis<0, N, T>> MakeVertex(const int &index) const {
        switch (index) {
            case 0: {
                return std::make_shared<PhyTensorBsplineBasis<0, N, T>>(this->_domain->AffineMap(
                        Coordinate(this->_domain->DomainStart(0), this->_domain->DomainStart(1))));
            }
            case 1: {
                return std::make_shared<PhyTensorBsplineBasis<0, N, T>>(this->_domain->AffineMap(
                        Coordinate(this->_domain->DomainEnd(0), this->_domain->DomainStart(1))));
            }
            case 2: {
                return std::make_shared<PhyTensorBsplineBasis<0, N, T>>(this->_domain->AffineMap(
                        Coordinate(this->_domain->DomainEnd(0), this->_domain->DomainEnd(1))));
            }
            case 3: {
                return std::make_shared<PhyTensorBsplineBasis<0, N, T>>(this->_domain->AffineMap(
                        Coordinate(this->_domain->DomainStart(0), this->_domain->DomainEnd(1))));
            }
        }
    }
};

#endif //OO_IGA_SURFACE_H
