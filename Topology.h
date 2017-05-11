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

    Element() : _domain(nullptr), _called(false) {};

    Element(DomainShared_ptr m) : _domain(m), _called(false) {};

    friend class Runner;

    bool BeCalled() const {
        return _called;
    }

    void Called() { _called = true; }

    virtual void accept(Visitor<T> &) = 0;

private:
    DomainShared_ptr _domain;
    bool _called;
};


enum Orientation {
    west, north, east, south
};

template<typename T>
class Edge : public Element<T> {
public:
    typedef typename Element<T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<T>::Coordinate Coordinate;

    Edge(const Orientation &orient = west) : Element<T>(), _position(orient), _matched(false), _pair(nullptr) {};

    Edge(DomainShared_ptr m, const Orientation &orient = west) : Element<T>(m), _position(orient), _matched(false), _pair(nullptr) {};

    void GetOrient() const {
        std::cout << _position << std::endl;
    }

    void accept(Visitor<T> &a) {};
private:
    Orientation _position;
    bool _matched;
    Coordinate _begin;
    Coordinate _end;
    Edge<T> *_pair;
};

#endif //OO_IGA_TOPOLOGY_H
