//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"
#include <array>
#include <algorithm>
#include "Utility.hpp"

template <int d, typename T>
class TensorBsplineBasis
{
  public:
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef std::vector<int> DiffPattern;

    typedef typename BsplineBasis<T>::BasisFunVal BasisFunVal;
    typedef typename BsplineBasis<T>::BasisFunValPac BasisFunValPac;
    typedef typename BsplineBasis<T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    typedef typename BsplineBasis<T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    using KnotSpan = std::pair<vector, vector>;
    using KnotSpanList = typename std::vector<KnotSpan>;

    TensorBsplineBasis();

    TensorBsplineBasis(const BsplineBasis<T> &baseX);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY, const BsplineBasis<T> &baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>> &KV);

    TensorBsplineBasis(const KnotVector<T> &kVX);

    TensorBsplineBasis(const KnotVector<T> &kVX, const KnotVector<T> &kVY);

    TensorBsplineBasis(const KnotVector<T> &kVX, const KnotVector<T> &kVY, const KnotVector<T> &kVZ);

    virtual ~TensorBsplineBasis(){};

    int
    GetDegree(const int i) const;

    //! Return the global d.o.f in this patch
    int
    GetDof() const;

    //! Return the d.o.f in i direction.
    int
    GetDof(const int i) const;

    // (TODO) there is a bug when p=1
    void
    BezierDualInitialize();

    //! Return the index in each directions.
    std::vector<int> TensorIndex(const int &m) const;

    int Index(const std::vector<int> &ts) const;

    std::unique_ptr<std::vector<int>> Indices() const;

    //! Return all index in given direction and given layer
    std::unique_ptr<std::vector<int>>
    HyperPlaneIndices(const int &, const int &) const;

    matrix Support(const int &i) const;

    int NumActive() const;

    int NumActive(const int &i) const;

    void ChangeKnots(const KnotVector<T> &, int);

    inline void PrintKnots(int i) const { _basis[i].PrintKnots(); }

    inline void PrintUniKnots(int i) const { _basis[i].PrintUniKnots(); }

    void PrintEvalTensor(const vector &u, const DiffPattern &diff = DiffPattern(d, 0)) const;

    void PrintEvalDerAllTensor(const vector &u, const int diff = 0) const;

    inline void KnotVectorSetter(const KnotVector<T> &knot_vector, int i)
    {
        _basis[i] = knot_vector;
    }

    inline const KnotVector<T> &KnotVectorGetter(int i) const
    {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].Knots();
    }

    inline T DomainStart(int i) const
    {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].DomainStart();
    }

    inline int MaxDegree() const
    {
        int res = 0;
        for (int i = 0; i < d; ++i)
        {
            res = std::max(res, GetDegree(i));
        }
        return res;
    }

    inline T DomainEnd(int i) const
    {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].DomainEnd();
    }

    virtual BasisFunValPac_ptr EvalTensor(const vector &u, const DiffPattern &i = DiffPattern(d, 0)) const;

    virtual BasisFunValDerAllList_ptr EvalDerAllTensor(const vector &u, const int i = 0) const;

    BasisFunValDerAllList_ptr EvalDualAllTensor(const vector &u) const;

    std::vector<int> ActiveIndex(const vector &u) const;

    T EvalSingle(const vector &u, const int n, const DiffPattern &i = DiffPattern(d, 0)) const;

    void KnotSpanGetter(KnotSpanList &) const;

    bool InDomain(const vector &u) const;

  protected:
    std::array<BsplineBasis<T>, d> _basis;

    TensorBsplineBasis(const TensorBsplineBasis<d, T> &) = delete;

    TensorBsplineBasis operator=(const TensorBsplineBasis<d, T> &) = delete;
};

template <typename T>
class TensorBsplineBasis<0, T>
{
  public:
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef std::vector<int> DiffPattern;

    typedef typename BsplineBasis<T>::BasisFunVal BasisFunVal;
    typedef typename BsplineBasis<T>::BasisFunValPac BasisFunValPac;
    typedef typename BsplineBasis<T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    typedef typename BsplineBasis<T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;

    TensorBsplineBasis();

    TensorBsplineBasis(const T &support);

    virtual ~TensorBsplineBasis(){};

    BasisFunValDerAllList_ptr
    EvalDualAllTensor(const vector &u) const
    {
        BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
        std::vector<T> result(1, 0);
        if (u(0) == _basis)
        {
            result[0] = 1;
        }
        Result->push_back(BasisFunValDerAll(0, result));
        return Result;
    }

    BasisFunValDerAllList_ptr
    EvalDerAllTensor(const vector &u, const int i = 0) const
    {
        BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
        std::vector<T> result(i + 1, 0);
        if (u(0) == _basis)
        {
            for (int j = 0; j < result.size(); ++j)
            {
                result[j] = 1;
            }
        }
        Result->push_back(BasisFunValDerAll(0, result));
        return Result;
    }

  protected:
    T _basis{0};
};

#endif //OO_IGA_TENSORBSPLINEBASIS_H
