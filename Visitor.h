//
// Created by miaodi on 08/06/2017.
//

#ifndef OO_IGA_VISITOR_H
#define OO_IGA_VISITOR_H

#include "Topology.h"
#include "DofMapper.h"
#include <thread>
#include <mutex>
#include <iomanip>

template<typename T>
class Element;

template<typename T>
class Edge;

template<typename T>
class Cell;

template<typename T>
class DofMapper;

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

    virtual void visit(Edge<T> *g) = 0;

    virtual void visit(Cell<T> *g) = 0;
};

template<typename T>
class PoissonMapperInitiator : public Visitor<T> {

public:
    PoissonMapperInitiator(DofMapper<T> &dofMap) : _dofMap(dofMap) {

    }

    void visit(Edge<T> *g) {
        if (!g->GetMatchInfo()) {
            auto tmp = g->AllActivatedDofsOfLayers(0);
            for (const auto &i:*tmp) {
                _dofMap.FreezedDofInserter(g->GetDomain(), i);
            }
        } else {
            if (!g->Slave()) return;
            auto tmp = g->AllActivatedDofsOfLayersExcept(0, 1);
            for (const auto &i:*tmp) {
                _dofMap.SlaveDofInserter(g->GetDomain(), i);
            }
        }

    }

    void visit(Cell<T> *g) {
        _dofMap.DomainLabel(g->GetDomain());
        _dofMap.PatchDofSetter(g->GetDomain(), g->GetDof());
    }

private:
    DofMapper<T> &_dofMap;
};

template<typename T>
class BiharmonicMapperInitiator : public Visitor<T> {

public:
    BiharmonicMapperInitiator(DofMapper<T> &dofMap) : _dofMap(dofMap) {

    }

    void visit(Edge<T> *g) {
        if (!g->GetMatchInfo()) {
            auto tmp = g->AllActivatedDofsOfLayers(1);
            for (const auto &i:*tmp) {
                _dofMap.FreezedDofInserter(g->GetDomain(), i);
            }
        } else {
            if (!g->Slave()) return;
            auto tmp = g->AllActivatedDofsOfLayersExcept(1, 2);
            for (const auto &i:*tmp) {
                _dofMap.SlaveDofInserter(g->GetDomain(), i);
            }
        }

    }

    void visit(Cell<T> *g) {
        _dofMap.DomainLabel(g->GetDomain());
        _dofMap.PatchDofSetter(g->GetDomain(), g->GetDof());
    }

private:
    DofMapper<T> &_dofMap;
};

template<typename T>
class PoissonVisitor : public Visitor<T> {
public:
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using LoadFunctor = typename Visitor<T>::LoadFunctor;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using CoordinatePair = typename QuadratureRule<T>::CoordinatePair;
public:
    PoissonVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : _bodyForceFunctor(load), _dofmap(dof) {

    }

    void visit(Cell<T> *g) {
        Initialize(g);
        Assemble(g);
    }

    void visit(Edge<T> *g) {

    }

    void Initialize(Element<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
    }

    void Assemble(Cell<T> *g) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        auto domain = g->GetDomain();

        std::mutex pmutex;
        auto n = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(n);
        const int grainsize = elements.size() / n;
        auto work_iter = elements.begin();


        auto lambda = [&](typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end) -> void {
            for (auto i = begin; i != end; ++i) {
                LocalAssemble(g, domain, *i, _bodyForceFunctor, pmutex);
            }
        };
        for (auto it = std::begin(threads); it != std::end(threads) - 1; ++it) {
            *it = std::thread(lambda, work_iter, work_iter + grainsize);
            work_iter += grainsize;
        }
        threads.back() = std::thread(lambda, work_iter, elements.end());
        for (auto &i:threads) {
            i.join();
        }
        this->_poissonStiffness.shrink_to_fit();
        this->_poissonBodyForce.shrink_to_fit();
    }

    virtual void
    LocalAssemble(Cell<T> *g, DomainShared_ptr const basis, const CoordinatePair &element, const LoadFunctor &load, std::mutex &pmutex) {
        Quadlist quadratures;
        this->_quadrature.MapToQuadrature(element, quadratures);
        auto initialIndex = _dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size() * 2);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size() * 2, index.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> weightsLoad(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunsLoad(quadratures.size(), index.size());
        int it = 0;
        for (const auto &i : quadratures) {
            auto evals = basis->Eval1DerAllTensor(i.first);
            weights(2 * it) = i.second * g->Jacobian(i.first);
            weights(2 * it + 1) = weights(2 * it);
            weightsLoad(it) = weights(2 * it) * load(basis->AffineMap(i.first))[0];///
            int itit = 0;
            for (const auto &j : *evals) {
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
                    std::lock_guard<std::mutex> lock(pmutex);
                    this->_poissonStiffness.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
            std::lock_guard<std::mutex> lock(pmutex);
            this->_poissonBodyForce.push_back(IndexedValue(index[i] + initialIndex, 0, tempLoadVector(i)));
        }
    }

    std::tuple<std::unique_ptr<Eigen::SparseMatrix<T>>, std::unique_ptr<Eigen::SparseMatrix<T>>> Domain() {
        auto triangleStiffnessMatrix = Accessory::SparseMatrixMaker<T>(_poissonStiffness);
        std::unique_ptr<Eigen::SparseMatrix<T>> stiffnessMatrix(new Eigen::SparseMatrix<T>);
        *stiffnessMatrix = triangleStiffnessMatrix->template selfadjointView<Eigen::Lower>();
        auto load = Accessory::SparseMatrixMaker<T>(_poissonBodyForce);
        return std::make_tuple(std::move(stiffnessMatrix), std::move(load));
    }

protected:
    IndexedValueList _poissonStiffness;
    IndexedValueList _poissonBodyForce;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _bodyForceFunctor;
};

template<typename T>
class BiharmonicVisitor : public PoissonVisitor<T> {
public:
    using DomainShared_ptr = typename PoissonVisitor<T>::DomainShared_ptr;
    using IndexedValue = typename PoissonVisitor<T>::IndexedValue;
    using IndexedValueList = typename PoissonVisitor<T>::IndexedValueList;
    using LoadFunctor = typename PoissonVisitor<T>::LoadFunctor;
    using CoordinatePairList = typename PoissonVisitor<T>::CoordinatePairList;
    using Quadlist = typename PoissonVisitor<T>::Quadlist;
    using CoordinatePair = typename PoissonVisitor<T>::CoordinatePair;
public:
    BiharmonicVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : PoissonVisitor<T>(dof, load) {

    }

    void
    LocalAssemble(Cell<T> *g, DomainShared_ptr const basis, const CoordinatePair &element, const LoadFunctor &load, std::mutex &pmutex) {
        Quadlist quadratures;
        this->_quadrature.MapToQuadrature(element, quadratures);
        auto initialIndex = this->_dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> weightsLoad(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunsLoad(quadratures.size(), index.size());
        int it = 0;
        for (const auto &i : quadratures) {
            auto evals = basis->Eval2DerAllTensor(i.first);
            weights(it) = i.second * g->Jacobian(i.first);

            weightsLoad(it) = weights(it) * load(basis->AffineMap(i.first))[0];///
            int itit = 0;
            for (const auto &j : *evals) {
                basisFuns(it, itit) = j.second[3] + j.second[5];
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
                    std::lock_guard<std::mutex> lock(pmutex);
                    this->_poissonStiffness.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
            std::lock_guard<std::mutex> lock(pmutex);
            this->_poissonBodyForce.push_back(IndexedValue(index[i] + initialIndex, 0, tempLoadVector(i)));
        }
    }
};


template<typename T>
class PoissonBoundaryVisitor : public Visitor<T> {
public:
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using LoadFunctor = typename Visitor<T>::LoadFunctor;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
public:
    PoissonBoundaryVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : _deformationFunctor(load), _dofmap(dof) {

    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (!g->GetMatchInfo()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
    }

    void Assemble(Edge<T> *g) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        Quadlist quadratures;
        auto domain = g->GetDomain();
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, quadratures, _deformationFunctor);
        }
        this->_poissonMass.shrink_to_fit();
        this->_poissonBoundary.shrink_to_fit();
    }

    virtual void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, const Quadlist &quadratures, const LoadFunctor &load) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> boundaryInfo(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        int it = 0;
        for (const auto &i : quadratures) {
            auto evals = basis->EvalDerAllTensor(i.first);
            weights(it) = i.second;
            boundaryInfo(it) = _deformationFunctor(basis->AffineMap(i.first))[0];
            int itit = 0;
            for (const auto &j : *evals) {
                basisFuns(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                basisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempLoadVector(
                basisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (i >= j) {
                    this->_poissonMass.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
            this->_poissonBoundary.push_back(IndexedValue(index[i] + initialIndex, 0, tempLoadVector(i)));
        }
    }

    std::unique_ptr<Eigen::SparseMatrix<T>> Boundary() {
        using namespace Eigen;
        auto a = Accessory::CondensedSparseMatrixMaker<T>(_poissonMass);
        auto b = Accessory::SparseMatrixGivenColRow<T>(std::get<0>(a), std::vector<int>{0}, _poissonBoundary);
        Eigen::SparseMatrix<T> Gramian;
        Gramian = std::get<2>(a)->template selfadjointView<Eigen::Lower>();
        Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
        solver.analyzePattern(Gramian);
        solver.factorize(Gramian);
        auto transform = Accessory::SparseTransform<T>(std::get<0>(a), _dofmap.Dof());
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        *result = transform->transpose() * solver.solve(*b);
        return result;

    }

protected:
    IndexedValueList _poissonMass;
    IndexedValueList _poissonBoundary;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _deformationFunctor;
};

template<typename T>
class BiharmonicBoundaryVisitor : public PoissonBoundaryVisitor<T> {
public:
    using DomainShared_ptr = typename PoissonBoundaryVisitor<T>::DomainShared_ptr;
    using IndexedValue = typename PoissonBoundaryVisitor<T>::IndexedValue;
    using IndexedValueList = typename PoissonBoundaryVisitor<T>::IndexedValueList;
    using LoadFunctor = typename PoissonBoundaryVisitor<T>::LoadFunctor;
    using Coordinate = typename Element<T>::Coordinate;
    using CoordinatePairList = typename PoissonBoundaryVisitor<T>::CoordinatePairList;
    using Quadlist = typename PoissonBoundaryVisitor<T>::Quadlist;
public:
    BiharmonicBoundaryVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : PoissonBoundaryVisitor<T>(dof, load) {

    }

    void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, const Quadlist &quadratures, const LoadFunctor &load) {
        auto initialIndex = this->_dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(2 * quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> boundaryInfo(2 * quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(2 * quadratures.size(), index.size());

        int it = 0;
        for (const auto &i : quadratures) {
            Coordinate phyPtr = basis->AffineMap(i.first);
            Coordinate normal = g->NormalDirection(i.first);
            auto evals = basis->Eval1DerAllTensor(i.first);
            weights(2 * it) = i.second;
            weights(2 * it + 1) = i.second;
            boundaryInfo(2 * it) = this->_deformationFunctor(phyPtr)[0];
            boundaryInfo(2 * it + 1) = this->_deformationFunctor(phyPtr)[1] * normal(0) + this->_deformationFunctor(phyPtr)[2] * normal(1);
            int itit = 0;
            for (const auto &j : *evals) {
                basisFuns(2 * it, itit) = j.second[0];
                basisFuns(2 * it + 1, itit) = j.second[1] * normal(0) + j.second[2] * normal(1);
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                basisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempLoadVector(
                basisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (i >= j) {
                    this->_poissonMass.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
            this->_poissonBoundary.push_back(IndexedValue(index[i] + initialIndex, 0, tempLoadVector(i)));
        }
    }
};

template<typename T>
class PoissonDGBoundaryVisitor : public Visitor<T> {
public:
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using LoadFunctor = typename Visitor<T>::LoadFunctor;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Coordinate = typename Visitor<T>::Coordinate;
public:
    PoissonDGBoundaryVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : _deformationFunctor(load), _dofmap(dof) {

    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (!g->GetMatchInfo()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
    }

    void Assemble(Edge<T> *g) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        Quadlist quadratures;
        auto domain = g->GetDomain();
        T h = g->Size() / elements.size();
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, h, quadratures);
        }
        this->_poissonMass.shrink_to_fit();
        this->_poissonBoundary.shrink_to_fit();
    }

    virtual void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, T h, const Quadlist &quadratures) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> boundaryInfo(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunctionTerm(quadratures.size(), index.size());
        int it = 0;
        T sigma = 1e3;
        for (const auto &i : quadratures) {
            auto evals = basis->Eval1DerAllTensor(i.first);
            auto Jac = g->Jacobian(i.first);
            Coordinate normal = g->NormalDirection(i.first);
            weights(it) = i.second * Jac;
            int itit = 0;
            for (const auto &j : *evals) {
                derivativeTerm(it, itit) = (j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = j.second[0];
                itit++;
            }
            boundaryInfo(it) = _deformationFunctor(basis->AffineMap(i.first))[0];
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempFlux;
        tempFlux =
                -derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFunctionTerm;
        tempFlux += tempFlux.transpose();
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempFluxLoad(
                -derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStablize;
        tempStablize =
                sigma / h * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFunctionTerm;
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempStablizeLoad(
                sigma / h * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        for (int i = 0; i != tempFlux.rows(); ++i) {
            for (int j = 0; j != tempFlux.cols(); ++j) {
                if (i >= j && tempFlux(i, j) != 0) {
                    this->_poissonMass.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempFlux(i, j)));
                }
            }
            this->_poissonBoundary.push_back(IndexedValue(index[i] + initialIndex, 0, tempFluxLoad(i)));
        }
        for (int i = 0; i != tempStablize.rows(); ++i) {
            for (int j = 0; j != tempStablize.cols(); ++j) {
                if (i >= j && tempStablize(i, j) != 0) {
                    this->_poissonMass.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStablize(i, j)));
                }
            }
            this->_poissonBoundary.push_back(IndexedValue(index[i] + initialIndex, 0, tempStablizeLoad(i)));
        }
    }

    std::tuple<std::unique_ptr<Eigen::SparseMatrix<T>>, std::unique_ptr<Eigen::SparseMatrix<T>>> Boundary() {
        auto dof = _dofmap.Dof();
        auto triangleStiffnessMatrix = Accessory::SparseMatrixMaker<T>(_poissonMass, dof, dof);
        std::unique_ptr<Eigen::SparseMatrix<T>> stiffnessMatrix(new Eigen::SparseMatrix<T>);
        *stiffnessMatrix = triangleStiffnessMatrix->template selfadjointView<Eigen::Lower>();
        auto load = Accessory::SparseMatrixMaker<T>(_poissonBoundary, dof, 1);
        return std::make_tuple(std::move(stiffnessMatrix), std::move(load));
    }

protected:
    IndexedValueList _poissonMass;
    IndexedValueList _poissonBoundary;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _deformationFunctor;
};

template<typename T>
class BiharmonicDGBoundaryVisitor : public Visitor<T> {
public:
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using LoadFunctor = typename Visitor<T>::LoadFunctor;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Coordinate = typename Visitor<T>::Coordinate;
public:
    BiharmonicDGBoundaryVisitor(const DofMapper<T> &dof, const LoadFunctor &load) : _deformationFunctor(load), _dofmap(dof) {

    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (!g->GetMatchInfo()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
    }

    void Assemble(Edge<T> *g) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        Quadlist quadratures;
        auto domain = g->GetDomain();
        T h = g->Size() / elements.size();
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, h, quadratures);
        }
        this->_poissonMass.shrink_to_fit();
        this->_poissonBoundary.shrink_to_fit();
    }

    virtual void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, T h, const Quadlist &quadratures) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        auto index = basis->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> boundaryInfo(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> boundaryInfoDer(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> thirdDerivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> LaplacianTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunctionTerm(quadratures.size(), index.size());
        int it = 0;
        T sigma = 1e4;
        for (const auto &i : quadratures) {
            auto evals = basis->Eval3DerAllTensor(i.first);
            auto Jac = g->Jacobian(i.first);
            Coordinate normal = g->NormalDirection(i.first);
            weights(it) = i.second * Jac;
            int itit = 0;
            Coordinate phyPtr = basis->AffineMap(i.first);
            for (const auto &j : *evals) {
                thirdDerivativeTerm(it, itit) = ((j.second[6] + j.second[8]) * normal(0) + (j.second[7] + j.second[9]) * normal(1));
                LaplacianTerm(it, itit) = j.second[3] + j.second[5];
                derivativeTerm(it, itit) = (j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = j.second[0];
                itit++;
            }
            boundaryInfo(it) = _deformationFunctor(phyPtr)[0];
            boundaryInfoDer(it) = this->_deformationFunctor(phyPtr)[1] * normal(0) + this->_deformationFunctor(phyPtr)[2] * normal(1);
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempFlux1, tempFlux2;
        tempFlux1 = basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) * thirdDerivativeTerm;
        tempFlux2 = -derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) * LaplacianTerm;
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempFlux1Load(
                thirdDerivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempFlux2Load(
                -LaplacianTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfoDer);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStablize1(
                sigma / pow(h, 3) * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFunctionTerm);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStablize2(
                sigma / h * derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                derivativeTerm);
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempStablizeLoad1(
                sigma / pow(h, 3) * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfo);
        Eigen::Matrix<T, Eigen::Dynamic, 1> tempStablizeLoad2(
                sigma / h * derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                boundaryInfoDer);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> stiffness =
                tempFlux1 + tempFlux1.transpose() + tempFlux2 + tempFlux2.transpose() + tempStablize1 + tempStablize2;
        Eigen::Matrix<T, Eigen::Dynamic, 1> load = tempFlux1Load + tempFlux2Load + tempStablizeLoad1 + tempStablizeLoad2;
        for (int i = 0; i != stiffness.rows(); ++i) {
            for (int j = 0; j != stiffness.cols(); ++j) {
                if (i >= j && stiffness(i, j) != 0) {
                    this->_poissonMass.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, stiffness(i, j)));
                }
            }
            this->_poissonBoundary.push_back(IndexedValue(index[i] + initialIndex, 0, load(i)));
        }
    }

    std::tuple<std::unique_ptr<Eigen::SparseMatrix<T>>, std::unique_ptr<Eigen::SparseMatrix<T>>> Boundary() {
        auto dof = _dofmap.Dof();
        auto triangleStiffnessMatrix = Accessory::SparseMatrixMaker<T>(_poissonMass, dof, dof);
        std::unique_ptr<Eigen::SparseMatrix<T>> stiffnessMatrix(new Eigen::SparseMatrix<T>);
        *stiffnessMatrix = triangleStiffnessMatrix->template selfadjointView<Eigen::Lower>();
        auto load = Accessory::SparseMatrixMaker<T>(_poissonBoundary, dof, 1);
        return std::make_tuple(std::move(stiffnessMatrix), std::move(load));
    }

protected:
    IndexedValueList _poissonMass;
    IndexedValueList _poissonBoundary;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _deformationFunctor;
};

template<typename T>
class PoissonInterfaceVisitor : public Visitor<T> {
public:
    using EdgeShared_Ptr = typename Element<T>::EdgeShared_Ptr;
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using Coordinate = typename Visitor<T>::Coordinate;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
public:
    PoissonInterfaceVisitor(const DofMapper<T> &dof) : _dofmap(dof) {
        int Dof = _dofmap.Dof();
        for (int i = 0; i != Dof; ++i) {
            _poissonInterface.push_back(IndexedValue(i, i, 1));
        }
    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (g->GetMatchInfo() && g->Slave()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        auto pairDeg_x = g->Counterpart()->GetDegree(0);
        auto pairDeg_y = g->Counterpart()->GetDegree(1);
        std::vector<int> degrees{deg_x, deg_y, pairDeg_x, pairDeg_y};
        this->_quadrature.SetUpQuadrature(*std::max_element(degrees.begin(), degrees.end()) + 1);
    }

    void Assemble(Edge<T> *g) {
        EdgeShared_Ptr lagrange = g->MakeEdge();
        EdgeShared_Ptr edgeDomain = g->Counterpart()->MakeEdge();
        auto multiplierKnots = lagrange->KnotVectorGetter(0);
        auto thisKnots = edgeDomain->KnotVectorGetter(0);
        auto thisUniKnots = thisKnots.GetUnique();
        Pts u, v;
        for (const auto &i:thisUniKnots) {
            u(0) = i;
            lagrange->InversePts(edgeDomain->AffineMap(u), v);
            multiplierKnots.Insert(v(0));
        }
        auto elements = multiplierKnots.KnotEigenSpans();
        Quadlist quadratures;
        auto domain = g->GetDomain();
        IndexedValueList matrixContainer;
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, lagrange, quadratures, matrixContainer);
            LocalAssemble(&*g->Counterpart(), g->Counterpart()->GetDomain(), lagrange, quadratures, matrixContainer);
        }
        SolveCouplingRelation(g, domain, matrixContainer);
        this->_poissonInterface.shrink_to_fit();
    }

    void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, EdgeShared_Ptr const lagrange,
                  const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        Coordinate u;
        basis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto index = basis->ActiveIndex(u);
        auto lagrangeIndex = lagrange->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lagrangeBasisFuns(quadratures.size(), lagrangeIndex.size());
        int it = 0;
        for (const auto &i : quadratures) {
            if (!basis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!g->IsOn(u)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            auto evals = basis->EvalDerAllTensor(u);
            auto lagrangeEvals = lagrange->EvalDerAllTensor(i.first);
            weights(it) = i.second;
            int itit = 0;
            for (const auto &j : *evals) {
                basisFuns(it, itit) = j.second[0];
                itit++;
            }
            itit = 0;
            for (const auto &j : *lagrangeEvals) {
                lagrangeBasisFuns(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                lagrangeBasisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (tempStiffMatrix(i, j) != 0) {
                    matrix.push_back(
                            IndexedValue(lagrangeIndex[i], index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
        }
    }

    void SolveCouplingRelation(Edge<T> *g, DomainShared_ptr const basis, IndexedValueList &matrix) {
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        auto index = Accessory::NonZeroCols<T>(matrix);
        auto row = Accessory::NonZeroRows<T>(matrix);
        auto slaveInDomain = _dofmap.SlaveDofIn(basis);
        auto slaveSideIndex = g->AllActivatedDofsOfLayers(0);
        int start = _dofmap.StartingIndex(basis);
        std::transform(slaveSideIndex->cbegin(), slaveSideIndex->cend(), slaveSideIndex->begin(), [&start](const int &i) { return i + start; });

        std::vector<int> slaveIndex, masterIndex, slaveMaster;
        std::set_intersection(index.begin(), index.end(), slaveInDomain.begin(), slaveInDomain.end(), std::back_inserter(slaveIndex));
        std::set_difference(index.begin(), index.end(), slaveInDomain.begin(), slaveInDomain.end(), std::back_inserter(masterIndex));
        std::set_difference(slaveSideIndex->begin(), slaveSideIndex->end(), slaveInDomain.begin(), slaveInDomain.end(),
                            std::back_inserter(slaveMaster));
        Matrix mass = *Accessory::SparseMatrixGivenColRow(row, slaveIndex, matrix);
        Matrix load = *Accessory::SparseMatrixGivenColRow(row, masterIndex, matrix);
        for (auto i:slaveMaster) {
            int j = std::find(masterIndex.begin(), masterIndex.end(), i) - masterIndex.begin();
            load.col(j) *= -1;
        }
        mass.row(1) = mass.row(1) + mass.row(0);
        mass.row(mass.rows() - 2) = mass.row(mass.rows() - 2) + mass.row(mass.rows() - 1);
        load.row(1) = load.row(1) + load.row(0);
        load.row(load.rows() - 2) = load.row(load.rows() - 2) + load.row(load.rows() - 1);
        Accessory::removeRow<T>(mass, 0);
        Accessory::removeRow<T>(mass, mass.rows() - 1);
        Accessory::removeRow<T>(load, 0);
        Accessory::removeRow<T>(load, load.rows() - 1);
        Matrix temp = mass.partialPivLu().solve(load);

        T tol = 1E-11;
        for (int i = 0; i != temp.rows(); ++i) {
            for (int j = 0; j != temp.cols(); ++j) {
                if (std::abs(temp(i, j)) > tol) {
                    _poissonInterface.push_back(
                            IndexedValue(masterIndex[j], slaveIndex[i], temp(i, j)));
                }
            }
        }
    }

    std::unique_ptr<Eigen::SparseMatrix<T>> Coupling() {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dofmap.Dof(), _dofmap.Dof());
        result->setFromTriplets(_poissonInterface.begin(), _poissonInterface.end());
        return result;
    }

private:
    IndexedValueList _poissonInterface;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
};

template<typename T>
class BiharmonicInterfaceVisitor : public Visitor<T> {
public:
    using EdgeShared_Ptr = typename Element<T>::EdgeShared_Ptr;
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using Coordinate = typename Visitor<T>::Coordinate;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
public:
    BiharmonicInterfaceVisitor(const DofMapper<T> &dof) : _dofmap(dof) {
        int Dof = _dofmap.Dof();
        for (int i = 0; i != Dof; ++i) {
            _poissonInterface.push_back(IndexedValue(i, i, 1));
        }
    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (g->GetMatchInfo() && g->Slave()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        auto pairDeg_x = g->Counterpart()->GetDegree(0);
        auto pairDeg_y = g->Counterpart()->GetDegree(1);
        std::vector<int> degrees{deg_x, deg_y, pairDeg_x, pairDeg_y};
        this->_quadrature.SetUpQuadrature(*std::max_element(degrees.begin(), degrees.end()) + 1);
    }

    void Assemble(Edge<T> *g) {
        EdgeShared_Ptr lagrange = g->MakeEdge();
        EdgeShared_Ptr edgeDomain = g->Counterpart()->MakeEdge();
        auto multiplierKnots = lagrange->KnotVectorGetter(0);
        auto thisKnots = edgeDomain->KnotVectorGetter(0);
        auto thisUniKnots = thisKnots.GetUnique();
        Pts u, v;
        for (const auto &i:thisUniKnots) {
            u(0) = i;
            lagrange->InversePts(edgeDomain->AffineMap(u), v);
            multiplierKnots.Insert(v(0));
        }
        auto elements = multiplierKnots.KnotEigenSpans();
        Quadlist quadratures;
        auto domain = g->GetDomain();
        IndexedValueList C0Container;
        IndexedValueList C1Container;
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            C0LocalAssemble(g, domain, lagrange, quadratures, C0Container);
            C0LocalAssemble(&*g->Counterpart(), g->Counterpart()->GetDomain(), lagrange, quadratures, C0Container);
            C1SlaveLocalAssemble(g, domain, lagrange, quadratures, C1Container);
            C1MasterLocalAssemble(&*g->Counterpart(), g->Counterpart()->GetDomain(), domain, lagrange, quadratures, C1Container);
        }
        SolveCouplingRelation(g, domain, C0Container, C1Container);
        this->_poissonInterface.shrink_to_fit();
    }

    void
    C0LocalAssemble(Edge<T> *g, DomainShared_ptr const basis, EdgeShared_Ptr const lagrange,
                    const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        Coordinate u;
        basis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto index = basis->ActiveIndex(u);
        auto lagrangeIndex = lagrange->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lagrangeBasisFuns(quadratures.size(), lagrangeIndex.size());
        int it = 0;
        for (const auto &i : quadratures) {
            if (!basis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!g->IsOn(u)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            auto evals = basis->EvalDerAllTensor(u);
            auto lagrangeEvals = lagrange->EvalDerAllTensor(i.first);
            weights(it) = i.second;
            int itit = 0;
            for (const auto &j : *evals) {
                basisFuns(it, itit) = j.second[0];
                itit++;
            }
            itit = 0;
            for (const auto &j : *lagrangeEvals) {
                lagrangeBasisFuns(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                lagrangeBasisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (tempStiffMatrix(i, j) != 0) {
                    matrix.push_back(
                            IndexedValue(lagrangeIndex[i], index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
        }
    }

    void
    C1SlaveLocalAssemble(Edge<T> *g, DomainShared_ptr const basis, EdgeShared_Ptr const lagrange,
                         const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        Coordinate u;
        basis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto index = basis->ActiveIndex(u);
        auto lagrangeIndex = lagrange->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lagrangeBasisFuns(quadratures.size(), lagrangeIndex.size());
        int it = 0;
        for (const auto &i : quadratures) {
            if (!basis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!g->IsOn(u)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            auto evals = basis->EvalDerAllTensor(u, 1);
            auto lagrangeEvals = lagrange->EvalDerAllTensor(i.first);

            weights(it) = i.second;
            int itit = 0;
            if (g->GetOrient() == 0 || g->GetOrient() == 2) {
                for (const auto &j : *evals) {
                    basisFuns(it, itit) = j.second[2];
                    itit++;
                }
            } else {
                for (const auto &j : *evals) {
                    basisFuns(it, itit) = j.second[1];
                    itit++;
                }
            }
            itit = 0;
            for (const auto &j : *lagrangeEvals) {
                lagrangeBasisFuns(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                lagrangeBasisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;
        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (tempStiffMatrix(i, j) != 0) {
                    matrix.push_back(
                            IndexedValue(lagrangeIndex[i], index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
        }
    }


    void
    C1MasterLocalAssemble(Edge<T> *g, DomainShared_ptr const basis, DomainShared_ptr const basisSlave, EdgeShared_Ptr const lagrange,
                          const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialIndex = _dofmap.StartingIndex(basis);
        Coordinate u;
        basis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto index = basis->ActiveIndex(u);
        auto lagrangeIndex = lagrange->ActiveIndex(quadratures[0].first);
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFuns(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lagrangeBasisFuns(quadratures.size(), lagrangeIndex.size());
        int it = 0;
        for (const auto &i : quadratures) {
            if (!basis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!g->IsOn(u)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            auto evals = basis->EvalDerAllTensor(u, 1);
            auto lagrangeEvals = lagrange->EvalDerAllTensor(i.first);

            Coordinate uSlave;
            basisSlave->InversePts(lagrange->AffineMap(i.first), uSlave);
            weights(it) = i.second;
            int itit = 0;
            T alpha, beta, gamma;
            if (g->Counterpart()->GetOrient() == 0 || g->Counterpart()->GetOrient() == 2) {
                auto geomDriXi = basis->AffineMap(u, {1, 0});
                auto geomDriEta = basis->AffineMap(u, {0, 1});
                auto geomDriEtaSlave = basisSlave->AffineMap(uSlave, {0, 1});
                alpha = geomDriEtaSlave(0) * geomDriXi(1) - geomDriXi(0) * geomDriEtaSlave(1);
                beta = geomDriEta(0) * geomDriEtaSlave(1) - geomDriEtaSlave(0) * geomDriEta(1);
                gamma = geomDriEta(0) * geomDriXi(1) - geomDriXi(0) * geomDriEta(1);
                for (const auto &j : *evals) {
                    basisFuns(it, itit) = j.second[1] * beta / gamma + j.second[2] * alpha / gamma;
                    itit++;
                }
            } else {
                auto geomDriXi = basis->AffineMap(u, {1, 0});
                auto geomDriEta = basis->AffineMap(u, {0, 1});
                auto geomDriXiSlave = basisSlave->AffineMap(uSlave, {1, 0});
                alpha = geomDriXiSlave(0) * geomDriXi(1) - geomDriXi(0) * geomDriXiSlave(1);
                beta = geomDriEta(0) * geomDriXiSlave(1) - geomDriXiSlave(0) * geomDriEta(1);
                gamma = geomDriEta(0) * geomDriXi(1) - geomDriXi(0) * geomDriEta(1);
                for (const auto &j : *evals) {
                    basisFuns(it, itit) = j.second[1] * beta / gamma + j.second[2] * alpha / gamma;
                    itit++;
                }
            }
            itit = 0;
            for (const auto &j : *lagrangeEvals) {
                lagrangeBasisFuns(it, itit) = j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStiffMatrix;
        tempStiffMatrix =
                lagrangeBasisFuns.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFuns;

        for (int i = 0; i != tempStiffMatrix.rows(); ++i) {
            for (int j = 0; j != tempStiffMatrix.cols(); ++j) {
                if (tempStiffMatrix(i, j) != 0) {
                    matrix.push_back(
                            IndexedValue(lagrangeIndex[i], index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
        }
    }

    void SolveCouplingRelation(Edge<T> *g, DomainShared_ptr const basis, IndexedValueList &C0matrix, IndexedValueList &C1matrix) {
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        auto C0index = Accessory::NonZeroCols<T>(C0matrix);
        auto C0row = Accessory::NonZeroRows<T>(C0matrix);
        auto slaveInDomain = _dofmap.SlaveDofIn(basis);
        auto C0slaveSideIndex = g->AllActivatedDofsOfLayers(0);
        int start = _dofmap.StartingIndex(basis);
        std::transform(C0slaveSideIndex->cbegin(), C0slaveSideIndex->cend(), C0slaveSideIndex->begin(), [&start](const int &i) { return i + start; });
        std::vector<int> C0slaveIndex, C0masterIndex, C0slaveMaster;
        std::set_intersection(C0index.begin(), C0index.end(), slaveInDomain.begin(), slaveInDomain.end(), std::back_inserter(C0slaveIndex));
        std::set_difference(C0index.begin(), C0index.end(), slaveInDomain.begin(), slaveInDomain.end(), std::back_inserter(C0masterIndex));
        std::set_difference(C0slaveSideIndex->begin(), C0slaveSideIndex->end(), slaveInDomain.begin(), slaveInDomain.end(),
                            std::back_inserter(C0slaveMaster));

        Matrix C0mass = *Accessory::SparseMatrixGivenColRow(C0row, C0slaveIndex, C0matrix);
        Matrix C0load = *Accessory::SparseMatrixGivenColRow(C0row, C0masterIndex, C0matrix);
        for (auto i:C0slaveMaster) {
            int j = std::find(C0masterIndex.begin(), C0masterIndex.end(), i) - C0masterIndex.begin();
            C0load.col(j) *= -1;
        }
        C0mass.row(2) = C0mass.row(2) + (C0mass.row(1) + C0mass.row(0));
        C0mass.row(C0mass.rows() - 3) = C0mass.row(C0mass.rows() - 3) + (C0mass.row(C0mass.rows() - 2) + C0mass.row(C0mass.rows() - 1));
        C0load.row(2) = C0load.row(2) + (C0load.row(1) + C0load.row(0));
        C0load.row(C0load.rows() - 3) = C0load.row(C0load.rows() - 3) + (C0load.row(C0load.rows() - 2) + C0load.row(C0load.rows() - 1));

        Accessory::removeRow<T>(C0mass, 0);
        Accessory::removeRow<T>(C0mass, 0);
        Accessory::removeRow<T>(C0mass, C0mass.rows() - 1);
        Accessory::removeRow<T>(C0mass, C0mass.rows() - 1);
        Accessory::removeRow<T>(C0load, 0);
        Accessory::removeRow<T>(C0load, 0);
        Accessory::removeRow<T>(C0load, C0load.rows() - 1);
        Accessory::removeRow<T>(C0load, C0load.rows() - 1);
        Matrix C0temp = C0mass.partialPivLu().solve(C0load);
        T tol = 1E-11;
        for (int i = 0; i != C0temp.rows(); ++i) {
            for (int j = 0; j != C0temp.cols(); ++j) {
                if (std::abs(C0temp(i, j)) > tol) {
                    _poissonInterface.push_back(
                            IndexedValue(C0masterIndex[j], C0slaveIndex[i], C0temp(i, j)));
                }
            }
        }

        auto C1index = Accessory::NonZeroCols<T>(C1matrix);
        auto C1row = Accessory::NonZeroRows<T>(C1matrix);
        auto C0C1slaveSideIndex = g->AllActivatedDofsOfLayers(1);
        std::transform(C0C1slaveSideIndex->cbegin(), C0C1slaveSideIndex->cend(), C0C1slaveSideIndex->begin(),
                       [&start](const int &i) { return i + start; });
        auto C1slaveSideIndex = std::make_shared<std::vector<int>>();
        std::set_difference(C0C1slaveSideIndex->begin(), C0C1slaveSideIndex->end(), C0slaveSideIndex->begin(), C0slaveSideIndex->end(),
                            std::back_inserter(*C1slaveSideIndex));
        std::vector<int> C1slaveIndex, C1masterIndex, C1slaveMaster;
        std::set_intersection(C1slaveSideIndex->begin(), C1slaveSideIndex->end(), slaveInDomain.begin(), slaveInDomain.end(),
                              std::back_inserter(C1slaveIndex));
        std::set_difference(C1index.begin(), C1index.end(), slaveInDomain.begin(), slaveInDomain.end(), std::back_inserter(C1masterIndex));
        std::set_difference(C0C1slaveSideIndex->begin(), C0C1slaveSideIndex->end(), C1slaveIndex.begin(), C1slaveIndex.end(),
                            std::back_inserter(C1slaveMaster));

        Matrix C1mass = *Accessory::SparseMatrixGivenColRow(C1row, C1slaveIndex, C1matrix);
        Matrix C1load = *Accessory::SparseMatrixGivenColRow(C1row, C1masterIndex, C1matrix);
        Matrix C1C0load = *Accessory::SparseMatrixGivenColRow(C1row, C0slaveIndex, C1matrix);
        for (auto i:C1slaveMaster) {
            auto position = std::find(C1masterIndex.begin(), C1masterIndex.end(), i);
            if (position != C1masterIndex.end()) {
                int j = position - C1masterIndex.begin();
                C1load.col(j) *= -1;
            }
        }

        for (auto i:C1slaveMaster) {
            auto position = std::find(C0slaveIndex.begin(), C0slaveIndex.end(), i);
            if (position != C0slaveIndex.end()) {
                int j = position - C0slaveIndex.begin();
                C1C0load.col(j) *= -1;
            }
        }

        C1mass.row(2) = C1mass.row(2) + (C1mass.row(1) + C1mass.row(0));
        C1mass.row(C1mass.rows() - 3) = C1mass.row(C1mass.rows() - 3) + (C1mass.row(C1mass.rows() - 2) + C1mass.row(C1mass.rows() - 1));
        C1load.row(2) = C1load.row(2) + (C1load.row(1) + C1load.row(0));
        C1load.row(C1load.rows() - 3) = C1load.row(C1load.rows() - 3) + (C1load.row(C1load.rows() - 2) + C1load.row(C1load.rows() - 1));
        C1C0load.row(2) = C1C0load.row(2) + (C1C0load.row(1) + C1C0load.row(0));
        C1C0load.row(C1C0load.rows() - 3) =
                C1C0load.row(C1C0load.rows() - 3) + (C1C0load.row(C1C0load.rows() - 2) + C1C0load.row(C1C0load.rows() - 1));
        Accessory::removeRow<T>(C1mass, 0);
        Accessory::removeRow<T>(C1mass, 0);
        Accessory::removeRow<T>(C1mass, C1mass.rows() - 1);
        Accessory::removeRow<T>(C1mass, C1mass.rows() - 1);
        Accessory::removeRow<T>(C1load, 0);
        Accessory::removeRow<T>(C1load, 0);
        Accessory::removeRow<T>(C1load, C1load.rows() - 1);
        Accessory::removeRow<T>(C1load, C1load.rows() - 1);
        Accessory::removeRow<T>(C1C0load, 0);
        Accessory::removeRow<T>(C1C0load, 0);
        Accessory::removeRow<T>(C1C0load, C1C0load.rows() - 1);
        Accessory::removeRow<T>(C1C0load, C1C0load.rows() - 1);
        Matrix C1temp = C1mass.partialPivLu().solve(C1load);
        Matrix C1C0temp = C1mass.partialPivLu().solve(C1C0load);
        Matrix temp = C1C0temp * C0temp;
        for (int i = 0; i != C1temp.rows(); ++i) {
            for (int j = 0; j != C1temp.cols(); ++j) {
                if (std::abs(C1temp(i, j)) > tol) {
                    _poissonInterface.push_back(
                            IndexedValue(C1masterIndex[j], C1slaveIndex[i], C1temp(i, j)));
                }
            }
        }
        for (int i = 0; i != temp.rows(); ++i) {
            for (int j = 0; j != temp.cols(); ++j) {
                if (std::abs(temp(i, j)) > tol) {
                    _poissonInterface.push_back(
                            IndexedValue(C0masterIndex[j], C1slaveIndex[i], temp(i, j)));
                }
            }
        }
    }

    std::unique_ptr<Eigen::SparseMatrix<T>> Coupling() {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dofmap.Dof(), _dofmap.Dof());
        result->setFromTriplets(_poissonInterface.begin(), _poissonInterface.end());
        return result;
    }

private:
    IndexedValueList _poissonInterface;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
};

template<typename T>
class PoissonDGInterfaceVisitor : public Visitor<T> {
public:
    using EdgeShared_Ptr = typename Element<T>::EdgeShared_Ptr;
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using Coordinate = typename Visitor<T>::Coordinate;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
public:
    PoissonDGInterfaceVisitor(const DofMapper<T> &dof) : _dofmap(dof) {
    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (g->GetMatchInfo() && g->Slave()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        auto pairDeg_x = g->Counterpart()->GetDegree(0);
        auto pairDeg_y = g->Counterpart()->GetDegree(1);
        std::vector<int> degrees{deg_x, deg_y, pairDeg_x, pairDeg_y};
        this->_quadrature.SetUpQuadrature(*std::max_element(degrees.begin(), degrees.end()) + 1);
    }

    void Assemble(Edge<T> *g) {
        EdgeShared_Ptr lagrange = g->MakeEdge();
        EdgeShared_Ptr edgeDomain = g->Counterpart()->MakeEdge();
        auto multiplierKnots = lagrange->KnotVectorGetter(0);
        auto thisKnots = edgeDomain->KnotVectorGetter(0);
        auto thisUniKnots = thisKnots.GetUnique();
        Pts u, v;
        for (const auto &i:thisUniKnots) {
            u(0) = i;
            lagrange->InversePts(edgeDomain->AffineMap(u), v);
            multiplierKnots.Insert(v(0));
        }
        auto elements = multiplierKnots.KnotEigenSpans();
        Quadlist quadratures;
        auto domain = g->GetDomain();
        T h = g->Size() / elements.size();
        IndexedValueList matrixContainer;
        for (const auto &i : elements) {
            Coordinate start, end;
            start = lagrange->AffineMap(i.first);
            end = lagrange->AffineMap(i.second);

            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, g->Counterpart()->GetDomain(), lagrange, h, quadratures, matrixContainer);
        }
        this->_poissonInterface.shrink_to_fit();
    }

    void
    LocalAssemble(Edge<T> *g, DomainShared_ptr const slaveBasis, DomainShared_ptr const masterBasis, EdgeShared_Ptr const lagrange,
                  T h, const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialSlaveIndex = _dofmap.StartingIndex(slaveBasis);
        auto initialMasterIndex = _dofmap.StartingIndex(masterBasis);
        Coordinate u, uM;
        slaveBasis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto slaveIndex = slaveBasis->ActiveIndex(u);
        masterBasis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto masterIndex = masterBasis->ActiveIndex(u);
        std::transform(slaveIndex.cbegin(), slaveIndex.cend(), slaveIndex.begin(),
                       [&initialSlaveIndex](const int &i) { return i + initialSlaveIndex; });
        std::transform(masterIndex.cbegin(), masterIndex.cend(), masterIndex.begin(),
                       [&initialMasterIndex](const int &i) { return i + initialMasterIndex; });
        auto index = slaveIndex;
        index.insert(index.end(), masterIndex.cbegin(), masterIndex.cend());
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunctionTerm(quadratures.size(), index.size());
        int it = 0;
        T sigma = 1e3;
        for (const auto &i : quadratures) {
            if (!slaveBasis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!masterBasis->InversePts(lagrange->AffineMap(i.first), uM)) {
                std::cout << "InversePts fail" << std::endl;
            }
            auto evals = slaveBasis->Eval1DerAllTensor(u);
            auto evalm = masterBasis->Eval1DerAllTensor(uM);
            auto Jac = g->Jacobian(u);
            Coordinate normal = g->NormalDirection(u);
            weights(it) = i.second * Jac;
            int itit = 0;
            for (const auto &j : *evals) {
                derivativeTerm(it, itit) = .5 * (j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = j.second[0];
                itit++;
            }
            for (const auto &j : *evalm) {
                derivativeTerm(it, itit) = .5 * (j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = -j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempFlux, tempStablize;
        tempFlux = -derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) * basisFunctionTerm;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempFluxSym = tempFlux + tempFlux.transpose();
        tempStablize = sigma / h * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                       basisFunctionTerm;
        for (int i = 0; i != tempFluxSym.rows(); ++i) {
            for (int j = 0; j != tempFluxSym.cols(); ++j) {
                if (tempFluxSym(i, j) != 0) {
                    _poissonInterface.push_back(
                            IndexedValue(index[i], index[j], tempFluxSym(i, j)));
                }
            }
        }
        for (int i = 0; i != tempStablize.rows(); ++i) {
            for (int j = 0; j != tempStablize.cols(); ++j) {
                if (tempStablize(i, j) != 0) {
                    _poissonInterface.push_back(
                            IndexedValue(index[i], index[j], tempStablize(i, j)));
                }
            }
        }
    }


    std::unique_ptr<Eigen::SparseMatrix<T>> DGInterface() {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dofmap.Dof(), _dofmap.Dof());
        result->setFromTriplets(_poissonInterface.begin(), _poissonInterface.end());
        return result;
    }

private:
    IndexedValueList _poissonInterface;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
};

template<typename T>
class BiharmonicDGInterfaceVisitor : public Visitor<T> {
public:
    using EdgeShared_Ptr = typename Element<T>::EdgeShared_Ptr;
    using DomainShared_ptr = typename Visitor<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;
    using Coordinate = typename Visitor<T>::Coordinate;
    using CoordinatePairList = typename Visitor<T>::CoordinatePairList;
    using Quadlist = typename Visitor<T>::Quadlist;
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
public:
    BiharmonicDGInterfaceVisitor(const DofMapper<T> &dof) : _dofmap(dof) {
    }

    void visit(Cell<T> *g) {
    }

    void visit(Edge<T> *g) {
        if (g->GetMatchInfo() && g->Slave()) {
            Initialize(g);
            Assemble(g);
        }
    }

    void Initialize(Edge<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        auto pairDeg_x = g->Counterpart()->GetDegree(0);
        auto pairDeg_y = g->Counterpart()->GetDegree(1);
        std::vector<int> degrees{deg_x, deg_y, pairDeg_x, pairDeg_y};
        this->_quadrature.SetUpQuadrature(*std::max_element(degrees.begin(), degrees.end()) + 1);
    }

    void Assemble(Edge<T> *g) {
        EdgeShared_Ptr lagrange = g->MakeEdge();
        EdgeShared_Ptr edgeDomain = g->Counterpart()->MakeEdge();
        auto multiplierKnots = lagrange->KnotVectorGetter(0);
        auto thisKnots = edgeDomain->KnotVectorGetter(0);
        auto thisUniKnots = thisKnots.GetUnique();
        Pts u, v;
        for (const auto &i:thisUniKnots) {
            u(0) = i;
            lagrange->InversePts(edgeDomain->AffineMap(u), v);
            multiplierKnots.Insert(v(0));
        }
        auto elements = multiplierKnots.KnotEigenSpans();
        Quadlist quadratures;
        auto domain = g->GetDomain();
        T h = g->Size() / elements.size();
        IndexedValueList matrixContainer;
        for (const auto &i : elements) {
            Coordinate start, end;
            start = lagrange->AffineMap(i.first);
            end = lagrange->AffineMap(i.second);

            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, &*g->Counterpart(), domain, g->Counterpart()->GetDomain(), lagrange, h, quadratures, matrixContainer);
        }
        this->_poissonInterface.shrink_to_fit();
    }

    void
    LocalAssemble(Edge<T> *g, Edge<T> *counterpart, DomainShared_ptr const slaveBasis, DomainShared_ptr const masterBasis, EdgeShared_Ptr const lagrange,
                  T h, const Quadlist &quadratures, IndexedValueList &matrix) {
        auto initialSlaveIndex = _dofmap.StartingIndex(slaveBasis);
        auto initialMasterIndex = _dofmap.StartingIndex(masterBasis);
        Coordinate u, uM;
        slaveBasis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto slaveIndex = slaveBasis->ActiveIndex(u);
        masterBasis->InversePts(lagrange->AffineMap(quadratures[0].first), u);
        auto masterIndex = masterBasis->ActiveIndex(u);
        std::transform(slaveIndex.cbegin(), slaveIndex.cend(), slaveIndex.begin(),
                       [&initialSlaveIndex](const int &i) { return i + initialSlaveIndex; });
        std::transform(masterIndex.cbegin(), masterIndex.cend(), masterIndex.begin(),
                       [&initialMasterIndex](const int &i) { return i + initialMasterIndex; });
        auto index = slaveIndex;
        index.insert(index.end(), masterIndex.cbegin(), masterIndex.cend());
        Eigen::Matrix<T, Eigen::Dynamic, 1> weights(quadratures.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> thirdDerivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> LaplacianTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derivativeTerm(quadratures.size(), index.size());
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basisFunctionTerm(quadratures.size(), index.size());
        int it = 0;
        T sigma = 1e2;
        for (const auto &i : quadratures) {
            if (!slaveBasis->InversePts(lagrange->AffineMap(i.first), u)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!masterBasis->InversePts(lagrange->AffineMap(i.first), uM)) {
                std::cout << "InversePts fail" << std::endl;
            }
            if (!g->IsOn(u)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            if (!counterpart->IsOn(uM)) {
                std::cout << "Gauss points is not on the edge" << std::endl;
            };
            auto evals = slaveBasis->Eval3DerAllTensor(u);
            auto evalm = masterBasis->Eval3DerAllTensor(uM);
            auto Jac = g->Jacobian(u);
            Coordinate normal = g->NormalDirection(u);
            weights(it) = i.second * Jac;
            int itit = 0;
            for (const auto &j : *evals) {
                thirdDerivativeTerm(it, itit) = .5 * ((j.second[6] + j.second[8]) * normal(0) + (j.second[7] + j.second[9]) * normal(1));
                LaplacianTerm(it, itit) = .5 * (j.second[3] + j.second[5]);
                derivativeTerm(it, itit) = (j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = j.second[0];
                itit++;
            }
            for (const auto &j : *evalm) {
                thirdDerivativeTerm(it, itit) = .5 * ((j.second[6] + j.second[8]) * normal(0) + (j.second[7] + j.second[9]) * normal(1));
                LaplacianTerm(it, itit) = .5 * (j.second[3] + j.second[5]);
                derivativeTerm(it, itit) = -(j.second[1] * normal(0) + j.second[2] * normal(1));
                basisFunctionTerm(it, itit) = -j.second[0];
                itit++;
            }
            it++;
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempFlux1, tempFlux2;
        tempFlux1 = basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) * thirdDerivativeTerm;

        tempFlux2 = -derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) * LaplacianTerm;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStablize1(
                sigma / pow(h, 3) * basisFunctionTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                basisFunctionTerm);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tempStablize2(
                sigma / h * derivativeTerm.transpose() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weights.asDiagonal()) *
                derivativeTerm);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> stiffness =
                tempFlux1 + tempFlux1.transpose() + tempFlux2 + tempFlux2.transpose() + tempStablize1 + tempStablize2;
        for (int i = 0; i != stiffness.rows(); ++i) {
            for (int j = 0; j != stiffness.cols(); ++j) {
                if (stiffness(i, j)!=0) {
                    _poissonInterface.push_back(
                            IndexedValue(index[i], index[j], stiffness(i, j)));
                }
            }
        }
    }


    std::unique_ptr<Eigen::SparseMatrix<T>> DGInterface() {
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result->resize(_dofmap.Dof(), _dofmap.Dof());
        result->setFromTriplets(_poissonInterface.begin(), _poissonInterface.end());
        return result;
    }

private:
    IndexedValueList _poissonInterface;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
};

#endif //OO_IGA_VISITOR_H
