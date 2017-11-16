//
// Created by di miao on 12/29/16.
//
#pragma once

#include <unordered_map>
#include "TensorBsplineBasis.h"
#include "Utility.hpp"
#include <eigen3/Eigen/StdVector>

template <int d, int N, typename T>
struct ComputeJacobian;

template <int d, int N, typename T = double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T>
{

  public:
    using Pts = Eigen::Matrix<T, d, 1>;
    using PhyPts = Eigen::Matrix<T, N, 1>;

    // aligned_allocator is required by Eigen for fixed-size Eigen types
    using GeometryVector = Accessory::ContPtsList<T, N>;
    typedef std::vector<int> DiffPattern;
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    using HyperPlane = PhyTensorBsplineBasis<d - 1, N, T>;
    using HyperPlaneSharedPts = std::shared_ptr<PhyTensorBsplineBasis<d - 1, N, T>>;

  public:
    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(const BsplineBasis<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &,
                          const BsplineBasis<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &,
                          const BsplineBasis<T> &,
                          const BsplineBasis<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const KnotVector<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const KnotVector<T> &,
                          const KnotVector<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const KnotVector<T> &,
                          const KnotVector<T> &,
                          const KnotVector<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &,
                          const Eigen::Matrix<T, Eigen::Dynamic, 1> &);

    virtual ~PhyTensorBsplineBasis()
    {
    }

    virtual PhyPts AffineMap(const Pts &,
                             const DiffPattern &i = DiffPattern(d, 0)) const;

    virtual T Jacobian(const Pts &) const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> JacobianMatrix(const Pts &) const;

    inline Pts Middle() const
    {
        Pts u;
        for (int i = 0; i != d; i++)
            u(i) = (this->_basis[i].DomainStart() + this->_basis[i].DomainEnd()) * .5;
        return u;
    }

    //! Return physical middle of the patch
    inline PhyPts PhyMiddle() const
    {
        return AffineMap(Middle());
    }

    bool InversePts(const PhyPts &,
                    Pts &,
                    int = 1e8,
                    T = std::numeric_limits<T>::epsilon() * 1e2) const;

    bool InversePts(const vector &,
                    vector &,
                    int = 1e8,
                    T = std::numeric_limits<T>::epsilon() * 1e2) const;

    virtual void DegreeElevate(int,
                               int);

    virtual void UniformRefine(int,
                               int,
                               int);

    void UniformRefineDof(int,
                          int);

    void KnotRefine(int,
                    const KnotVector<T> &);

    virtual void KnotInsertion(int,
                               T,
                               int = 1);

    inline virtual void DegreeElevate(int p)
    {
        if (p == 0)
            return;
        for (int i = 0; i != d; ++i)
            DegreeElevate(i, p);
    }

    inline virtual void UniformRefine(int r, int m)
    {
        if (r == 0)
            return;
        for (int i = 0; i != d; ++i)
            UniformRefine(i, r, m);
    }

    virtual void UniformRefineDof(int dof);

    void PrintCtrPts() const;

    inline PhyPts CtrPtsGetter(const int &i) const
    {
        return _geometricInfo[i];
    }

    inline void CtrPtsSetter(const int &i, const PhyPts &pt)
    {
        ASSERT(i < _geometricInfo.size(), "The control point index is out of range.\n");
        _geometricInfo[i] = pt;
    }

    HyperPlaneSharedPts MakeHyperPlane(const int &orientation,
                                       const int &layer) const;

    // Only defined for 2D domain represented by 2D parametric domain
    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval1PhyDerAllTensor(const vector &u) const;

    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval2PhyDerAllTensor(const vector &u) const;

    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval3PhyDerAllTensor(const vector &u) const;

  protected:
    GeometryVector _geometricInfo;
};

template <>
PhyTensorBsplineBasis<2, 1, double>::PhyTensorBsplineBasis(const std::vector<KnotVector<double>> &base,
                                                           const Eigen::Matrix<double, Eigen::Dynamic, 1> &geometry)
    : TensorBsplineBasis<2, double>(
          base)
{
    ASSERT(geometry.rows() == (this->TensorBsplineBasis<2, double>::GetDof()),
           "Invalid geometrical information input, check size bro.");
    for (int i = 0; i != geometry.rows(); ++i)
    {
        _geometricInfo.push_back(Eigen::Matrix<double, 1, 1>(geometry(i)));
    }
}

template <int N, typename T>
class PhyTensorBsplineBasis<0, N, T> : public TensorBsplineBasis<0, T>
{
  public:
    using PhyPts = Eigen::Matrix<T, N, 1>;
    typedef std::vector<int> DiffPattern;

    typedef typename BsplineBasis<T>::BasisFunVal BasisFunVal;
    typedef typename BsplineBasis<T>::BasisFunValPac BasisFunValPac;
    typedef typename BsplineBasis<T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    typedef typename BsplineBasis<T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;

    PhyTensorBsplineBasis(const T &knot, const PhyPts &point);

    PhyTensorBsplineBasis(const PhyPts &point);

    ~PhyTensorBsplineBasis(){};

    PhyPts
    Position() const
    {
        return _point;
    }

  protected:
    PhyPts _point;
};

#ifndef PHYTENSORBSPLINEBASIS_HPP
#include "../src/PhyTensorBsplineBasis.hpp"
#endif //