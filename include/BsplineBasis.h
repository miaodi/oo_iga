//
// Created by miaodi on 25/12/2016.
//

#pragma once

#include "KnotVector.h"
#include <memory>
#include "Utility.hpp"
#include "QuadratureRule.h"

template <typename T>
class BsplineBasis
{
  public:
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using block = Eigen::Block<matrix>;

    typedef std::pair<int, T> BasisFunVal;
    typedef std::vector<BasisFunVal> BasisFunValPac;
    typedef std::unique_ptr<BasisFunValPac> BasisFunValPac_ptr;
    using BasisFunValDerAll = std::pair<int, std::vector<T>>;
    using BasisFunValDerAllList = std::vector<BasisFunValDerAll>;
    using BasisFunValDerAllList_ptr = std::unique_ptr<BasisFunValDerAllList>;

    BsplineBasis();

    BsplineBasis(KnotVector<T> target);

    int GetDegree() const;

    int GetDof() const;

    int FindSpan(const T &u) const;

    BasisFunValDerAllList_ptr EvalDerAll(const T &u, int i) const;

    BasisFunValPac_ptr Eval(const T &u, const int i = 0) const;

    T EvalSingle(const T &u, const int n, const int i = 0) const;

    vector Support(const int i) const;

    vector InSpan(const T &u) const;

    inline T DomainStart() const { return _basisKnot[GetDegree()]; }

    inline T DomainEnd() const { return _basisKnot[GetDof()]; }

    int NumActive() const { return GetDegree() + 1; }

    inline const KnotVector<T> &Knots() const
    {
        return _basisKnot;
    }

    inline bool InDomain(T const &u) const { return ((u >= DomainStart()) && (u <= DomainEnd())); }

    inline void PrintKnots() const { _basisKnot.printKnotVector(); }

    inline void PrintUniKnots() const { _basisKnot.printUnique(); }

    inline int FirstActive(T u) const
    {
        return (InDomain(u) ? FindSpan(u) - GetDegree() : 0);
    }

    std::unique_ptr<matrix> BasisWeight() const;

    bool IsActive(const int i, const T u) const;

    void BezierDualInitialize();

    // Reduce the order of first two and last two elements by one (Serve as the Lagrange multiplier). The weights for boundary basis are computed. (Only C^{p-1} spline are considered.)
    void ModifyBoundaryInitialize();

    // Return the evaluation of the modified b-spline basis functions.
    BasisFunValDerAllList_ptr EvalModifiedDerAll(const T &u, int i) const;

    BasisFunValDerAllList_ptr BezierDual(const T &u) const;

  protected:
    KnotVector<T> _basisKnot;
    std::vector<matrix> _reconstruction;
    matrix _basisWeight;
    matrix _gramianInv;
};
