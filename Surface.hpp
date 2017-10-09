//
// Created by di miao on 10/9/17.
//

#ifndef OO_IGA_SURFACE_H
#define OO_IGA_SURFACE_H

#include "Topology.hpp"
#include "Edge.hpp"
#include "Visitor.hpp"

template<int N, typename T>
class Edge;

template<int N, typename T>
class Surface : public Element<2, N, T> {
public:
    typedef typename Element<2, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<2, N, T>::Coordinate Coordinate;
    typedef typename Element<2, N, T>::CoordinatePairList CoordinatePairList;

    Surface() : Element<2, N, T>() {};

    Surface(DomainShared_ptr m) : Element<2, N, T>(m) {
        for (int i = south; i <= west; ++i) {
            const Orientation orient = static_cast<Orientation>(i);
            _edges[i] = std::make_shared<Edge<N, T>>(orient);
        }
    }


    void Accept(Visitor<2, N, T> &a) {
        a.Visit(this);
    };

    void EdgeAccept(Visitor<1, N, T> &a) {
        for (auto &i:_edges) {
            i->Accept(a);
        }
    };


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

    T Size() const {
        return 0;
    }

    void Match(std::shared_ptr<Surface<N, T>> &counterpart) {
        for (auto &i:_edges) {
            for (auto &j:counterpart->_edges) {
                i->Match(j);
            }

        }
    }

    void PrintEdgeInfo() const {
        for (const auto &i:_edges) {
            i->PrintInfo();
        }
        /*
        for (const auto &i:_edges) {
            std::cout << "Starting Point: " << "(" << i->GetStartCoordinate()(0) << "," << i->GetStartCoordinate()(1)
                      << ")" << "   ";
            std::cout << "Ending Point: " << "(" << i->GetEndCoordinate()(0) << "," << i->GetEndCoordinate()(1)
                      << ")" << "   ";
            std::cout << "Matched?: " << i->GetMatchInfo() << std::endl;
        }
         */
        std::cout << std::endl;
    }

    T Jacobian(const Coordinate &u) const {
        return this->_domain->Jacobian(u);
    }

protected:
    std::array<std::shared_ptr<Edge<N, T>>, 4> _edges;
};

#endif //OO_IGA_SURFACE_H
