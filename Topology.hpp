//
// Created by miaodi on 10/05/2017.
//

#ifndef OO_IGA_TOPOLOGY_H
#define OO_IGA_TOPOLOGY_H

#include "PhyTensorBsplineBasis.h"
#include <eigen3/Eigen/Sparse>
#include "Visitor.hpp"

template<int d, int N, typename T>
class Visitor;


template<int d, int N, typename T>
class Element {
public:
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<d, N, T>>;

    //! Coordinate in the parametric domain.
    using Coordinate = Eigen::Matrix<T, d, 1>;
    using PhyPts = Eigen::Matrix<T, N, 1>;
    using LoadFunctor = std::function<std::vector<T>(const Coordinate &)>;
    using CoordinatePairList = typename std::vector<std::pair<Coordinate, Coordinate>>;


    Element();

    Element(const DomainShared_ptr &m);

    //! Area for surface, length for Edge, 0 for Vertex
    virtual T Measure() const = 0;

    virtual void Accept(Visitor<d, N, T> &) = 0;

    virtual std::unique_ptr<std::vector<int>> Indices(const int &) const = 0;

    virtual std::unique_ptr<std::vector<int>> ExclusiveIndices(const int &) const = 0;

    virtual void PrintIndices(const int &layerNum) const {
        auto res = Indices(layerNum);
        for (const auto &i:*res) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    bool BeCalled() const {
        return _called;
    }

    DomainShared_ptr GetDomain() const {
        return _domain;
    }

    void Called() { _called = true; }

protected:
    DomainShared_ptr _domain;
    bool _called;
};

template<int d, int N, typename T>
Element<d, N, T>::Element():_domain{std::make_shared<PhyTensorBsplineBasis<d, N, T>>()}, _called{false} {

}

template<int d, int N, typename T>
Element<d, N, T>::Element(const Element::DomainShared_ptr &m):_domain{m}, _called{false} {

}

enum Orientation {
    south = 0, east, north, west
};

/*


template<typename T>
class Edge : public Element<T>, public std::enable_shared_from_this<Edge<T>> {
public:
    typedef typename Element<T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<T>::Coordinate Coordinate;
    typedef typename Element<T>::CoordinatePairList CoordinatePairList;
    using PhyPts = typename PhyTensorBsplineBasis<2, 2, T>::PhyPts;
    using LoadFunctor = typename Element<T>::LoadFunctor;
    using EdgeShared_Ptr = typename Element<T>::EdgeShared_Ptr;

    Edge(const Orientation &orient = west)
            : Element<T>(), _position(orient), _matched(false), _pair(nullptr) {};

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

    Coordinate GetEndCoordinate() const {
        return _end;
    }

    T Size() const {
        return sqrt(pow(_begin(0) - _end(0), 2) + pow(_begin(1) - _end(1), 2));
    }

    T Jacobian(const Coordinate &u) const {
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
        return sqrt(pow(derivative(0), 2) + pow(derivative(1), 2));
    } //need specific define.

    bool GetMatchInfo() const {
        return _matched;
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

    bool Match(std::shared_ptr<Edge<T>> counterpart) {
        if (_matched == true || counterpart->_matched == true) {
            return true;
        }
        if (((_begin == counterpart->_begin) && (_end == counterpart->_end)) ||
            ((_begin == counterpart->_end) && (_end == counterpart->_begin))) {
            _pair = counterpart;
            _matched = true;
            counterpart->_pair = this->shared_from_this();
            counterpart->_matched = true;
            if (this->GetDof() > _pair->GetDof()) {
                _slave = true;
            } else {
                _pair->_slave = true;
            }
            return true;
        }
        return false;
    }

    bool Slave() const {
        return _slave;
    }

    bool IsOn(Coordinate &u) const {
        T tol = 1e-10;
        auto knot_x_begin = this->_domain->DomainStart(0);
        auto knot_x_end = this->_domain->DomainEnd(0);
        auto knot_y_begin = this->_domain->DomainStart(1);
        auto knot_y_end = this->_domain->DomainEnd(1);

        switch (_position) {
            case west: {
                if ((std::abs(u(0) - knot_x_begin)) < tol &&
                    (u(1) >= knot_y_begin - tol && u(1) <= knot_y_end + tol)) {
                    u(0) = knot_x_begin;
                    return true;
                }
                return false;
            }
            case east: {
                if ((std::abs(u(0) - knot_x_end)) < tol &&
                    (u(1) >= knot_y_begin - tol && u(1) <= knot_y_end + tol)) {
                    u(0) = knot_x_end;
                    return true;
                }
                return false;
            }
            case north: {
                if ((std::abs(u(1) - knot_y_end)) < tol &&
                    (u(0) >= knot_x_begin - tol && u(0) <= knot_x_end + tol)) {
                    u(1) = knot_y_end;
                    return true;
                }
                return false;
            }
            case south: {
                if ((std::abs(u(1) - knot_y_begin)) < tol &&
                    (u(0) >= knot_x_begin - tol && u(0) <= knot_x_end + tol)) {
                    u(1) = knot_y_begin;
                    return true;
                }
                return false;
            }
        }
    }

    void accept(Visitor<T> &a) {
        a.visit(this);
    };

    int GetDof() const {
        return MakeEdge()->GetDof();
    }

    EdgeShared_Ptr MakeEdge() const {
        switch (_position) {
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

    bool InversePts(const PhyPts &point, T &knotCoordinate) const {
        Coordinate pt;
        if (!this->_domain->InversePts(point, pt)) return false;
        if (IsOn(pt)) {
            switch (_position) {
                case west: {
                    knotCoordinate = pt(1);
                    return true;
                }
                case east: {
                    knotCoordinate = pt(1);
                    return true;
                }
                case north: {
                    knotCoordinate = pt(0);
                    return true;
                }
                case south: {
                    knotCoordinate = pt(0);
                    return true;
                }
            }
        }
        return false;
    }

    std::shared_ptr<Edge<T>> Counterpart() const {
        return _pair;
    }

    std::unique_ptr<std::vector<int>> AllActivatedDofsOfLayersExcept(const int &layerNum, const int &exceptNum) {
        std::unique_ptr<std::vector<int>> res(new std::vector<int>);
        switch (_position) {
            case west: {
                for (int i = 0; i <= layerNum; ++i) {
                    auto tmp = this->_domain->AllActivatedDofsOnBoundary(0, i);
                    res->insert(res->end(), tmp->begin() + exceptNum, tmp->end() - exceptNum);
                }
                break;
            }
            case east: {
                for (int i = 0; i <= layerNum; ++i) {
                    auto tmp = this->_domain->AllActivatedDofsOnBoundary(0, this->_domain->GetDof(0) - 1 - i);
                    res->insert(res->end(), tmp->begin() + exceptNum, tmp->end() - exceptNum);
                }
                break;
            }
            case north: {
                for (int i = 0; i <= layerNum; ++i) {
                    auto tmp = this->_domain->AllActivatedDofsOnBoundary(1, this->_domain->GetDof(1) - 1 - i);
                    res->insert(res->end(), tmp->begin() + exceptNum, tmp->end() - exceptNum);
                }
                break;
            }
            case south: {
                for (int i = 0; i <= layerNum; ++i) {
                    auto tmp = this->_domain->AllActivatedDofsOnBoundary(1, i);
                    res->insert(res->end(), tmp->begin() + exceptNum, tmp->end() - exceptNum);
                }
                break;
            }
        }
        return res;
    }

    //Given the layer number, it returns a pointer to a vector that contains all control points index of that layer.
    std::unique_ptr<std::vector<int>> AllActivatedDofsOfLayers(const int &layerNum) {
        return AllActivatedDofsOfLayersExcept(layerNum, 0);
    }

    void PrintActivatedDofsOfLayers(const int &layerNum) {
        auto res = AllActivatedDofsOfLayers(layerNum);
        std::cout << "Activated Dofs on this edge are:";
        for (const auto &i:*res) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

private:
    Orientation _position;
    bool _matched;
    bool _slave = false;
    Coordinate _begin;
    Coordinate _end;
    std::shared_ptr<Edge<T>> _pair;

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
    using LoadFunctor = typename Visitor<T>::LoadFunctor;

    Cell() : Element<T>() {};

    Cell(DomainShared_ptr m) : Element<T>(m) {
        for (int i = south; i <= west; ++i) {
            _edges[i] = std::make_shared<Edge<T>>(this->_domain, static_cast<Orientation>(i));
        }
    }


    void accept(Visitor<T> &a) {
        a.visit(this);
        for (int i = 0; i < 4; i++) {
            _edges[i]->accept(a);
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

    void Match(std::shared_ptr<Cell<T>> counterpart) {
        for (auto &i:_edges) {
            for (auto &j:counterpart->_edges) {
                i->Match(j);
            }

        }
    }

    void PrintEdgeInfo() const {
        for (const auto &i:_edges) {
            std::cout << "Starting Point: " << "(" << i->GetStartCoordinate()(0) << "," << i->GetStartCoordinate()(1)
                      << ")" << "   ";
            std::cout << "Ending Point: " << "(" << i->GetEndCoordinate()(0) << "," << i->GetEndCoordinate()(1)
                      << ")" << "   ";
            std::cout << "Matched?: " << i->GetMatchInfo() << std::endl;
        }
        std::cout << std::endl;
    }

    T Jacobian(const Coordinate &u) const {
        return this->_domain->Jacobian(u);
    }

public:
    std::array<std::shared_ptr<Edge<T>>, 4> _edges;
};

*/
#endif //OO_IGA_TOPOLOGY_H
