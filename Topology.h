//
// Created by miaodi on 10/05/2017.
//

#ifndef OO_IGA_TOPOLOGY_H
#define OO_IGA_TOPOLOGY_H

#include "PhyTensorBsplineBasis.h"

template<typename T>
class Visitor;

class Runner;

template<typename T>
class Element {
public:
    typedef typename std::shared_ptr<PhyTensorBsplineBasis<2, 2, T>> DomainShared_ptr;
    using Coordinate = Eigen::Matrix<T, 2, 1>;
    typedef std::vector<std::pair<Coordinate, Coordinate>> CoordinatePairList;

    Element() : _domain(nullptr), _called(false) {};

    Element(DomainShared_ptr m) : _domain(m), _called(false) {};

    friend class Runner;

    bool BeCalled() const {
        return _called;
    }

    void Called() { _called = true; }

    virtual void accept(Visitor<T> &) = 0;

    virtual void KnotSpansGetter(CoordinatePairList &) = 0;

protected:
    DomainShared_ptr _domain;
    bool _called;
};


enum Orientation {
    west = 0, north, east, south
};

template<typename T>
class Edge : public Element<T> {
public:
    typedef typename Element<T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<T>::Coordinate Coordinate;
    typedef typename Element<T>::CoordinatePairList CoordinatePairList;

    Edge(const Orientation &orient = west)
            : Element<T>(), _position(orient), _matched(false), _pair(nullptr) { VertexSetter(); };

    Edge(DomainShared_ptr m, const Orientation &orient = west) : Element<T>(m), _position(orient), _matched(false),
                                                                 _pair(nullptr) { VertexSetter(); };

    void PrintOrient() const {
        std::cout << _position << std::endl;
    }

    void PrintStartCoordinate() const {
        std::cout << _begin.transpose() << std::endl;
    }

    void PrintEndCoordinate() const {
        std::cout << _end.transpose() << std::endl;
    }

    Orientation GetOrient() const {
        return _position;
    }

    Coordinate GetStartCoordinate() const {
        return _begin;
    }

    Coordinate EndCoordinate() const {
        return _end;
    }

    void KnotSpansGetter(CoordinatePairList &knotspanslist) {
        switch (_position) {
            case west: {
                auto knot_y = this->_domain->KnotVectorGetter(1);
                auto knot_x = this->_domain->DomainStart(0);
                auto knotspan_y = knot_y.KnotSpans();
                knotspanslist.reserve(knotspan_y.size());
                for (const auto &i:knotspan_y) {
                    Coordinate _begin;
                    _begin << knot_x, i.first;
                    Coordinate _end;
                    _end << knot_x, i.second;
                    knotspanslist.push_back({_begin, _end});
                }
                break;
            }
            case east: {
                auto knot_y = this->_domain->KnotVectorGetter(1);
                auto knot_x = this->_domain->DomainEnd(0);
                auto knotspan_y = knot_y.KnotSpans();
                knotspanslist.reserve(knotspan_y.size());
                for (const auto &i:knotspan_y) {
                    Coordinate _begin;
                    _begin << knot_x, i.first;
                    Coordinate _end;
                    _end << knot_x, i.second;
                    knotspanslist.push_back({_begin, _end});
                }
                break;
            }
            case south: {
                auto knot_y = this->_domain->DomainStart(1);
                auto knot_x = this->_domain->KnotVectorGetter(0);
                auto knotspan_x = knot_x.KnotSpans();
                knotspanslist.reserve(knotspan_x.size());
                for (const auto &i:knotspan_x) {
                    Coordinate _begin;
                    _begin << i.first, knot_y;
                    Coordinate _end;
                    _end << i.second, knot_y;
                    knotspanslist.push_back({_begin, _end});
                }
                break;
            }
            case north: {
                auto knot_y = this->_domain->DomainEnd(1);
                auto knot_x = this->_domain->KnotVectorGetter(0);
                auto knotspan_x = knot_x.KnotSpans();
                knotspanslist.reserve(knotspan_x.size());
                for (const auto &i:knotspan_x) {
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

    Coordinate NormalDirection(const Coordinate &u) const {
        Coordinate derivative;
        switch (_position) {
            case west: {
                derivative = -this->_domain->AffineMap(u, {0, 1});
                break;
            }
            case east: {
                derivative = this->_domain->AffineMap(u, {0, 1});
                break;
            }
            case north: {
                derivative = -this->_domain->AffineMap(u, {1, 0});
                break;
            }
            case south: {
                derivative = this->_domain->AffineMap(u, {1, 0});
                break;
            }
        }
        Coordinate candidate1, candidate2;
        candidate1 << derivative(1), -derivative(0);
        candidate2 << -derivative(1), derivative(0);
        Eigen::Matrix<T, 2, 2> tmp;
        tmp.col(1) = derivative;
        tmp.col(0) = candidate1;
        if (tmp.determinant() > 0) {
            return 1.0 / candidate1.norm() * candidate1;
        } else {
            return 1.0 / candidate2.norm() * candidate2;
        }
    }

    void accept(Visitor<T> &a) {};
private:
    Orientation _position;
    bool _matched;
    Coordinate _begin;
    Coordinate _end;
    Edge<T> *_pair;

    void VertexSetter() {
        switch (_position) {
            case west: {
                Coordinate m, n;
                m << this->_domain->DomainStart(0), this->_domain->DomainEnd(1);
                n << this->_domain->DomainStart(0), this->_domain->DomainStart(1);
                _begin = this->_domain->AffineMap(m);
                _end = this->_domain->AffineMap(n);
                break;
            }
            case east: {
                Coordinate m, n;
                m << this->_domain->DomainEnd(0), this->_domain->DomainStart(1);
                n << this->_domain->DomainEnd(0), this->_domain->DomainEnd(1);
                _begin = this->_domain->AffineMap(m);
                _end = this->_domain->AffineMap(n);
                break;
            }
            case north: {
                Coordinate m, n;
                m << this->_domain->DomainEnd(0), this->_domain->DomainEnd(1);
                n << this->_domain->DomainStart(0), this->_domain->DomainEnd(1);
                _begin = this->_domain->AffineMap(m);
                _end = this->_domain->AffineMap(n);
                break;
            }
            case south: {
                Coordinate m, n;
                m << this->_domain->DomainStart(0), this->_domain->DomainStart(1);
                n << this->_domain->DomainEnd(0), this->_domain->DomainStart(1);
                _begin = this->_domain->AffineMap(m);
                _end = this->_domain->AffineMap(n);
                break;
            }
        }
    }
};

template<typename T>
class Cell : public Element<T> {
public:
    typedef typename Element<T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<T>::Coordinate Coordinate;
    typedef typename Element<T>::CoordinatePairList CoordinatePairList;

    Cell() : Element<T>() {};

    Cell(DomainShared_ptr m) : Element<T>(m) {
        for (int i = west; i <= south; ++i) {
            _edges[i] = Edge<T>(this->_domain, static_cast<Orientation>(i));
            std::cout<<i<<std::endl;
        }
    }//???

    void accept(Visitor<T> &a) {};

    void KnotSpansGetter(CoordinatePairList &) {};
    std::array<Edge<T>, 4> _edges;
};

#endif //OO_IGA_TOPOLOGY_H
