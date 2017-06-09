//
// Created by miaodi on 08/06/2017.
//

#ifndef OO_IGA_VISITOR_H
#define OO_IGA_VISITOR_H

#include "Topology.h"


template<typename T>
class Element;

template<typename T>
class Edge;

template<typename T>
class Cell;

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

    virtual void Initialize(Element<T> *g) {
        auto dof = g->GetDof();
        auto deg_x = g->GetDegree(0);
        auto deg_y = g->GetDegree(1);
        this->_quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
        this->_matrixList.reserve(dof * dof * _quadrature.NumOfQuadrature() / 3);
    }

    virtual void Initialize(Edge<T> *, Edge<T> *) =0;

    virtual void Assemble(Element<T> *, DomainShared_ptr const, const LoadFunctor &)=0;

    virtual void Assemble()=0;


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


protected:

    QuadratureRule<T> _quadrature;
    IndexedValueList _matrixList;
};


#endif //OO_IGA_VISITOR_H
