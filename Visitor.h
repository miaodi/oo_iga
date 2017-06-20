//
// Created by miaodi on 08/06/2017.
//

#ifndef OO_IGA_VISITOR_H
#define OO_IGA_VISITOR_H

#include "Topology.h"
#include "DofMapper.h"
#include <boost/bimap.hpp>

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
        if (g->BeCalled()) return;
        if (!g->GetMatchInfo()) {
            auto tmp = g->AllActivatedDofsOfLayers(0);
            for (const auto &i:*tmp) {
                _dofMap.FreezedDofInserter(g->GetDomain(), i);
            }
            g->Called();
        }else{
            
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
        Quadlist quadratures;
        IndexedValueList tempList;
        auto domain = g->GetDomain();
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, quadratures, _bodyForceFunctor);
        }
        this->_poissonStiffness.shrink_to_fit();
        this->_poissonBodyForce.shrink_to_fit();
    }

    void
    LocalAssemble(Cell<T> *g, DomainShared_ptr const basis, const Quadlist &quadratures, const LoadFunctor &load) {
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
                    this->_poissonStiffness.push_back(
                            IndexedValue(index[i] + initialIndex, index[j] + initialIndex, tempStiffMatrix(i, j)));
                }
            }
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

private:
    IndexedValueList _poissonStiffness;
    IndexedValueList _poissonBodyForce;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _bodyForceFunctor;
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

    void Initialize(Element<T> *g) {
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
    }

    void Assemble(Element<T> *g) {
        CoordinatePairList elements;
        g->KnotSpansGetter(elements);
        Quadlist quadratures;
        IndexedValueList tempList;
        auto domain = g->GetDomain();
        for (const auto &i : elements) {
            this->_quadrature.MapToQuadrature(i, quadratures);
            LocalAssemble(g, domain, quadratures, _deformationFunctor);
        }
        this->_poissonMass.shrink_to_fit();
        this->_poissonBoundary.shrink_to_fit();
    }

    void
    LocalAssemble(Element<T> *g, DomainShared_ptr const basis, const Quadlist &quadratures, const LoadFunctor &load) {
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
        auto a = Accessory::CondensedSparseMatrixMaker<T>(_poissonMass);
        auto b = Accessory::SparseMatrixGivenColRow<T>(std::get<0>(a), std::vector<int>{0}, _poissonBoundary);
        Eigen::SparseMatrix<T> Gramian;
        Gramian = std::get<2>(a)->template selfadjointView<Eigen::Lower>();
        Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower | Eigen::Upper> cg;
        cg.compute(Gramian);
        auto transform = Accessory::SparseTransform<T>(std::get<0>(a), _dofmap.Dof());
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        *result = transform->transpose() * cg.solve(*b);
        return result;

    }

private:
    IndexedValueList _poissonMass;
    IndexedValueList _poissonBoundary;
    const DofMapper<T> &_dofmap;
    QuadratureRule<T> _quadrature;
    LoadFunctor _deformationFunctor;
};

#endif //OO_IGA_VISITOR_H
