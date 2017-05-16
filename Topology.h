//
// Created by miaodi on 10/05/2017.
//

#ifndef OO_IGA_TOPOLOGY_H
#define OO_IGA_TOPOLOGY_H

#include "PhyTensorBsplineBasis.h"
#include "QuadratureRule.h"
#include <eigen3/Eigen/Sparse>
#include "MmpMatrix.h"

template<typename T>
class Visitor;

class Runner;

template<typename T>
class Element {
public:
    typedef typename std::shared_ptr<PhyTensorBsplineBasis<2, 2, T>> DomainShared_ptr;
    using Coordinate = Eigen::Matrix<T, 2, 1>;
    using LoadFunctor = std::function<std::vector<T>(const Coordinate &)>;
    using CoordinatePairList = typename std::vector<std::pair<Coordinate, Coordinate>>;

    Element() : _domain(std::make_shared<PhyTensorBsplineBasis<2, 2, T>>()), _called(false) {};

    Element(DomainShared_ptr m) : _domain(m), _called(false) {};

    friend class Runner;

    bool BeCalled() const {
        return _called;
    }

    void Called() { _called = true; }

    virtual void accept(Visitor<T> &, const LoadFunctor &) = 0;

    virtual void KnotSpansGetter(CoordinatePairList &) = 0;

    virtual T Jacobian(const Coordinate &) const =0;

    virtual int GetDof() const {
        return _domain->GetDof();
    }

    int GetDegree(const int i) const {
        return _domain->GetDegree(i);
    }

protected:
    DomainShared_ptr _domain;
    bool _called;
};


enum Orientation {
    south = 0, east, north, west
};

template<typename T>
class Edge : public Element<T>, public std::enable_shared_from_this<Edge<T>> {
public:
    typedef typename Element<T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<T>::Coordinate Coordinate;
    typedef typename Element<T>::CoordinatePairList CoordinatePairList;
    using LoadFunctor = typename Element<T>::LoadFunctor;

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

    T Jacobian(const Coordinate &u) const { return 0; } //need specific define.

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
            return true;
        }
        return false;
    }

    void accept(Visitor<T> &a, const LoadFunctor &functor) {};
private:
    Orientation _position;
    bool _matched;
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

    void accept(Visitor<T> &a, const LoadFunctor &functor) {
        a.Initialize(this);
        a.Assemble(this, this->_domain, functor);
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

    void Match(std::shared_ptr<Cell<T>> counterpart) {
        for (auto &i:_edges) {
            for (auto &j:counterpart->_edges) {
                if (i->Match(j) == true) break;
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

    std::array<std::shared_ptr<Edge<T>>, 4> _edges;
};

template<typename T>
class Visitor {
public:
    using Coordinate = typename Element<T>::Coordinate;
    using CoordinatePairList = typename Element<T>::CoordinatePairList;
    using Quadlist = typename QuadratureRule<T>::QuadList;
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;
    using IndexedValue = Eigen::Triplet<T>;
    using IndexedValueList = std::vector<IndexedValue>;
    using LoadFunctor = std::function<std::vector<T>(const Coordinate &)>;


    Visitor() {};

    void Initialize(Element<T> *g) {
        _dof = g->GetDof();
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
        this->_globalStiffMatrix.reserve(_dof * _dof * _quadrature.NumOfQuadrature()/3);
        this->_globalLoadVector.reserve(_dof * _quadrature.NumOfQuadrature());
    }

    void Assemble(Element<T> *g, DomainShared_ptr basis, const LoadFunctor &loadFun) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        Quadlist quadratures;
        IndexedValueList tempList;
        for (const auto &i:elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, basis, quadratures, loadFun);
        }
        this->_globalStiffMatrix.shrink_to_fit();
        this->_globalLoadVector.shrink_to_fit();
    }

    std::unique_ptr<Eigen::SparseMatrix<T>> MakeSparseMatrix() const {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dof, _dof);
        result->setFromTriplets(_globalStiffMatrix.begin(), _globalStiffMatrix.end());
        return result;
    }

    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> MakeDenseMatrix() const {
        auto sparse = MakeSparseMatrix();
        std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> result(
                new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(*sparse));
        return result;
    }

    std::unique_ptr<Eigen::SparseMatrix<T>> MakeSparseVector() const {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dof, 1);
        result->setFromTriplets(_globalLoadVector.begin(), _globalLoadVector.end());
        return result;
    }

    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1>> MakeDenseVector() const {
        auto sparse = MakeSparseVector();
        std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1>> result(new Eigen::Matrix<T, Eigen::Dynamic, 1>(*sparse));
        return result;
    }

protected:
    virtual void LocalAssemble(Element<T> *, DomainShared_ptr, const Quadlist &, const LoadFunctor &) = 0;

    QuadratureRule<T> _quadrature;
    IndexedValueList _globalStiffMatrix;
    IndexedValueList _globalLoadVector;
    int _dof;
};

template<typename T>
class PoissonVisitor : public Visitor<T> {
public:
    using Quadlist = typename Visitor<T>::Quadlist;
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;
    using IndexedValue = Eigen::Triplet<T>;
    using LoadFunctor = typename Visitor<T>::LoadFunctor;

    PoissonVisitor() : Visitor<T>() {};
protected:
    void LocalAssemble(Element<T> *g, DomainShared_ptr basis, const Quadlist &quadratures, const LoadFunctor &load) {
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size() * 2);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size() * 2, index.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> weightsLoad(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunsLoad(quadratures.size(), index.size());
        int it = 0;
        for (const auto &i:quadratures) {
            auto evals = basis->Eval1DerAllTensor(i.first);
            weights(2 * it) = i.second * g->Jacobian(i.first);
            weights(2 * it + 1) = weights(2 * it);
            weightsLoad(it) = weights(2 * it) * load(basis->AffineMap(i.first))[0];///
            int itit = 0;
            for (const auto &j:*evals) {
                basisFuns(2 * it, itit) = j.second[1];
                basisFuns(2 * it + 1, itit) = j.second[2];
                basisFunsLoad(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                basisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempLoadVector(basisFunsLoad.transpose() * weightsLoad);
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (i >= j) {
                    this->_globalStiffMatrix.push_back(IndexedValue(index[i], index[j], tempStiffMatrix(i, j)));
                }
            }
            this->_globalLoadVector.push_back(IndexedValue(index[i], 0, tempLoadVector(i)));
        }
    }
};

#endif //OO_IGA_TOPOLOGY_H
