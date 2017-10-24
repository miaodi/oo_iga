//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"
#include <array>
#include <algorithm>
#include "Utility.hpp"


template<int d, typename T = double>
class TensorBsplineBasis {
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

    TensorBsplineBasis(const BsplineBasis<T>& baseX);

    TensorBsplineBasis(const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY);

    TensorBsplineBasis(const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY, const BsplineBasis<T>& baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>>& KV);

    virtual ~TensorBsplineBasis() { };

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
    BezierDualInitialize()
    {
        for (int direction = 0; direction<d; ++direction)
        {
            _basis[direction].BezierDualInitialize();
        }
    }

    //! Return the index in each directions.
    std::vector<int>
    TensorIndex(const int& m) const
    {
        ASSERT(m<GetDof(), "Input index is invalid.");
        std::vector<int> ind(d);
        int mm = m;
        ///int always >=0.
        for (int i = static_cast<int>(d-1); i>=0; --i)
        {
            ind[i] = mm%GetDof(i);
            mm -= ind[i];
            mm /= GetDof(i);
        }
        return ind;
    }

    int
    Index(const std::vector<int>& ts) const
    {
        ASSERT(ts.size()==d, "Input index is invalid.");
        int index = 0;
        for (int direction = 0; direction<d; ++direction)
        {
            index += ts[direction];
            if (direction!=d-1)
            {
                index *= GetDof(direction+1);
            }
        }
        return index;
    }

    std::unique_ptr<std::vector<int>>
    Indices() const;

    //! Return all index in given direction and given layer
    std::unique_ptr<std::vector<int>>
    HyperPlaneIndices(const int&, const int&) const;

    matrix
    Support(const int& i) const
    {
        matrix res(static_cast<int>(d), 2);
        auto ti = TensorIndex(i);
        for (int j = 0; j!=d; ++j)
            res.row(j) = _basis[j].Support(ti[j]);
        return res;
    }

    int
    NumActive() const;

    int
    NumActive(const int& i) const;

    void
    ChangeKnots(const KnotVector<T>&, int);

    void
    PrintKnots(int i) const { _basis[i].PrintKnots(); }

    void
    PrintUniKnots(int i) const { _basis[i].PrintUniKnots(); }

    void
    PrintEvalTensor(const vector& u, const DiffPattern& diff = DiffPattern(d, 0)) const;

    void
    PrintEvalDerAllTensor(const vector& u, const int diff = 0) const;

    const KnotVector<T>&
    KnotVectorGetter(int i) const
    {
        ASSERT(i<d, "Invalid dimension index provided.");
        return _basis[i].Knots();
    }

    T
    DomainStart(int i) const
    {
        ASSERT(i<d, "Invalid dimension index provided.");
        return _basis[i].DomainStart();
    }

    int
    MaxDegree() const
    {
        int res = 0;
        for (int i = 0; i<d; ++i)
        {
            res = std::max(res, GetDegree(i));
        }
        return res;
    }

    T
    DomainEnd(int i) const
    {
        ASSERT(i<d, "Invalid dimension index provided.");
        return _basis[i].DomainEnd();
    }

    virtual BasisFunValPac_ptr
    EvalTensor(const vector& u, const DiffPattern& i = DiffPattern(d, 0)) const;

    virtual BasisFunValDerAllList_ptr
    EvalDerAllTensor(const vector& u, const int i = 0) const;

    BasisFunValDerAllList_ptr
    EvalDualAllTensor(const vector& u) const;

    std::vector<int>
    ActiveIndex(const vector& u) const
    {
        std::vector<int> temp;
        temp.reserve(NumActive());
        ASSERT((u.size()==d), "Invalid input vector size.");
        std::vector<int> indexes(d, 0);
        std::vector<int> endPerIndex(d);
        std::vector<int> startIndex(d);
        for (int i = 0; i!=d; ++i)
        {
            startIndex[i] = _basis[i].FirstActive(u(i));
            endPerIndex[i] = _basis[i].NumActive();
        }
        std::function<void(std::vector<int>&, const std::vector<int>&, int)> recursive;
        std::vector<int> multiIndex(d);
        recursive = [this, &startIndex, &temp, &multiIndex, &recursive](
                std::vector<int>& indexes,
                const std::vector<int>& endPerIndex,
                int direction)
        {
            if (direction==indexes.size())
            {
                temp.push_back(Index(multiIndex));
            }
            else
            {
                for (indexes[direction] = 0; indexes[direction]!=endPerIndex[direction]; indexes[direction]++)
                {
                    multiIndex[direction] = startIndex[direction]+indexes[direction];
                    recursive(indexes, endPerIndex, direction+1);
                }
            }
        };
        recursive(indexes, endPerIndex, 0);
        return temp;
    }

    T
    EvalSingle(const vector& u, const int n, const TensorBsplineBasis::DiffPattern& i)
    {
        ASSERT((u.size()==d) && (i.size()==d), "Invalid input vector size.");
        auto tensorindex = TensorIndex(n);
        T result = 1;
        for (int direction = 0; direction!=d; ++direction)
        {
            result *= _basis[direction].EvalSingle(u(direction), tensorindex[direction], i[direction]);
        }
        return result;
    }

    void
    KnotSpanGetter(KnotSpanList&) const;

    bool InDomain(const vector& u) const;

protected:
    std::array<BsplineBasis<T>, d> _basis;

    TensorBsplineBasis(const TensorBsplineBasis<d, T>&) { };

    TensorBsplineBasis
    operator=(const TensorBsplineBasis<d, T>&) { };
};

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis()
{
    for (int i = 0; i<d; ++i)
        _basis[i] = BsplineBasis<T>();
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T>& baseX)
{
    ASSERT(d==1, "Invalid dimension.");
    _basis[0] = baseX;
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY)
{
    ASSERT(d==2, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T>& baseX, const BsplineBasis<T>& baseY,
        const BsplineBasis<T>& baseZ)
{
    ASSERT(d==3, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const std::vector<KnotVector<T>>& knotVectors)
{
    ASSERT(d==knotVectors.size(), "Invalid number of knot-vectors given.");
    for (int i = 0; i!=d; ++i)
        _basis[i] = BsplineBasis<T>(knotVectors[i]);
}

template<int d, typename T>
int
TensorBsplineBasis<d, T>::GetDegree(const int i) const
{
    return _basis[i].GetDegree();
}

template<int d, typename T>
int
TensorBsplineBasis<d, T>::GetDof() const
{
    int dof = 1;
    for (int i = 0; i!=d; ++i)
        dof *= _basis[i].GetDof();
    return dof;
}

template<int d, typename T>
int
TensorBsplineBasis<d, T>::GetDof(const int i) const
{
    return _basis[i].GetDof();
}

template<int d, typename T>
int
TensorBsplineBasis<d, T>::NumActive(const int& i) const
{
    return _basis[i].NumActive();
}

template<int d, typename T>
int
TensorBsplineBasis<d, T>::NumActive() const
{
    int active = 1;
    for (int i = 0; i!=d; ++i)
        active *= _basis[i].NumActive();
    return active;
}

template<int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr
TensorBsplineBasis<d, T>::EvalTensor(const TensorBsplineBasis::vector& u,
        const TensorBsplineBasis::DiffPattern& i) const
{
    ASSERT((u.size()==d) && (i.size()==d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    std::array<BasisFunValPac_ptr, d> OneDResult;
    for (int direction = 0; direction!=d; ++direction)
    {
        OneDResult[direction] = _basis[direction].Eval(u(direction), i[direction]);
        endPerIndex.push_back(OneDResult[direction]->size());
    }
    std::vector<int> MultiIndex(d);
    std::vector<T> Value(d);
    BasisFunValPac_ptr Result(new BasisFunValPac);

    std::function<void(std::vector<int>&, const std::vector<int>&, int)> recursive;

    recursive = [this, &OneDResult, &MultiIndex, &Value, &Result, &recursive](std::vector<int>& indexes,
            const std::vector<int>& endPerIndex,
            int direction)
    {
        if (direction==indexes.size())
        {
            T result = 1;
            for (int ii = 0; ii<d; ii++)
                result *= Value[ii];
            Result->push_back(BasisFunVal(Index(MultiIndex), result));
        }
        else
        {
            for (indexes[direction] = 0; indexes[direction]!=endPerIndex[direction]; indexes[direction]++)
            {
                Value[direction] = (*OneDResult[direction])[indexes[direction]].second;
                MultiIndex[direction] = (*OneDResult[direction])[indexes[direction]].first;
                recursive(indexes, endPerIndex, direction+1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    return Result;
}

template<int d, typename T>
void
TensorBsplineBasis<d, T>::ChangeKnots(const KnotVector<T>& knots, int direction)
{
    _basis[direction] = knots;
}

template<int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr
TensorBsplineBasis<d, T>::EvalDualAllTensor(const TensorBsplineBasis::vector& u) const
{
    ASSERT((u.size()==d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    std::vector<int> MultiIndex(d);
    std::vector<T> Value(d);
    BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
    std::array<BasisFunValDerAllList_ptr, d> oneDResult;
    for (int direction = 0; direction!=d; ++direction)
    {
        oneDResult[direction] = _basis[direction].BezierDual(static_cast<T>(u(direction)));
        endPerIndex.push_back(oneDResult[direction]->size());
    }
    std::function<void(std::vector<int>&, const std::vector<int>&, int)> recursive;
    recursive = [this, &oneDResult, &MultiIndex, &Value, &Result, &recursive](std::vector<int>& indexes,
            const std::vector<int>& endPerIndex,
            int direction)
    {
        if (direction==indexes.size())
        {
            std::vector<T> result(1, 1);
            for (int ii = 0; ii<d; ii++)
                result[0] *= Value[ii];
            Result->push_back(BasisFunValDerAll(Index(MultiIndex), result));
        }
        else
        {
            for (indexes[direction] = 0; indexes[direction]!=endPerIndex[direction]; indexes[direction]++)
            {
                Value[direction] = (*oneDResult[direction])[indexes[direction]].second[0];
                MultiIndex[direction] = (*oneDResult[direction])[indexes[direction]].first;
                recursive(indexes, endPerIndex, direction+1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    return Result;
}

template<int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr
TensorBsplineBasis<d, T>::EvalDerAllTensor(const TensorBsplineBasis::vector& u,
        const int i) const
{
    ASSERT((u.size()==d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    Accessory::DifferentialPatternList differentialPatternList;
    for (int order = 0; order<=i; ++order)
    {
        auto temp = Accessory::PartialDerPattern<d>(order);
        differentialPatternList.insert(differentialPatternList.end(), temp->begin(), temp->end());
    }
    int derivativeAmount = differentialPatternList.size();
    BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
    std::array<BasisFunValDerAllList_ptr, d> oneDResult;
    for (int direction = 0; direction!=d; ++direction)
    {
        oneDResult[direction] = _basis[direction].EvalDerAll(u(direction), i);
        endPerIndex.push_back(oneDResult[direction]->size());
    }
    std::function<void(std::vector<int>&, const std::vector<int>&, int)> recursive;
    std::vector<int> multiIndex(d);
    std::vector<std::vector<T>> Values(derivativeAmount, std::vector<T>(d, 0));
    recursive =
            [this, &derivativeAmount, &oneDResult, &multiIndex, &Values, &Result, &differentialPatternList, &recursive](
                    std::vector<int>& indexes,
                    const std::vector<int>& endPerIndex,
                    int direction)
            {
                if (direction==indexes.size())
                {
                    std::vector<T> result(derivativeAmount, 1);
                    for (int iii = 0; iii!=derivativeAmount; ++iii)
                    {
                        for (int ii = 0; ii!=d; ++ii)
                        {
                            result[iii] *= Values[iii][ii];
                        }
                    }
                    Result->push_back(BasisFunValDerAll(Index(multiIndex), result));
                }
                else
                {
                    for (indexes[direction] = 0; indexes[direction]!=endPerIndex[direction]; indexes[direction]++)
                    {
                        for (auto it_diffPart = differentialPatternList.begin();
                             it_diffPart!=differentialPatternList.end(); ++it_diffPart)
                        {
                            int diffPart_label = it_diffPart-differentialPatternList.begin();
                            Values[diffPart_label][direction] =
                                    (*oneDResult[direction])[indexes[direction]].second[(*it_diffPart)[direction]];
                        }
                        multiIndex[direction] = (*oneDResult[direction])[indexes[direction]].first;
                        recursive(indexes, endPerIndex, direction+1);
                    }
                }
            };
    recursive(indexes, endPerIndex, 0);
    return Result;
}

//orientation is the normal direction,
template<int d, typename T>
std::unique_ptr<std::vector<int>>
TensorBsplineBasis<d, T>::HyperPlaneIndices(const int& orientation, const int& layer) const
{
    ASSERT(orientation<d, "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex(d, 0);
    for (int i = 0; i!=d; ++i)
    {
        if (i==orientation)
        {
            endPerIndex[i] = 1;
        }
        else
        {
            endPerIndex[i] = GetDof(i);
        }
    }
    std::unique_ptr<std::vector<int>> result(new std::vector<int>);
    std::function<void(std::vector<int>&, const std::vector<int>&, int)> recursive;
    std::vector<int> temp(d, 0);
    recursive = [this, &orientation, &layer, &result, &temp, &recursive](std::vector<int>& indexes,
            const std::vector<int>& endPerIndex,
            int direction)
    {
        if (direction==d)
        {
            result->push_back(Index(temp));
        }
        else
        {
            if (direction==orientation)
            {
                temp[direction] = layer;
                recursive(indexes, endPerIndex, direction+1);
            }
            else
            {
                for (indexes[direction] = 0; indexes[direction]!=endPerIndex[direction]; indexes[direction]++)
                {
                    temp[direction] = indexes[direction];
                    recursive(indexes, endPerIndex, direction+1);
                }
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    return result;
}

template<int d, typename T>
void
TensorBsplineBasis<d, T>::PrintEvalDerAllTensor(const TensorBsplineBasis::vector& u, const int diff) const
{
    auto eval = EvalDerAllTensor(u, diff);
    for (const auto& i : *eval)
    {
        std::cout << i.first << "th basis: ";
        for (const auto& j : i.second)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

template<int d, typename T>
void
TensorBsplineBasis<d, T>::PrintEvalTensor(const TensorBsplineBasis::vector& u,
        const TensorBsplineBasis::DiffPattern& diff) const
{
    auto eval = EvalTensor(u, diff);
    for (const auto& i : *eval)
    {
        std::cout << i.first << "th basis: " << i.second << std::endl;
    }
}

template<int d, typename T>
std::unique_ptr<std::vector<int>>
TensorBsplineBasis<d, T>::Indices() const
{
    auto dof = this->GetDof();
    auto res = std::make_unique<std::vector<int>>();
    for (int i = 0; i<dof; ++i)
    {
        res->push_back(i);
    }
    return res;
}

// Create knot pairs that represent the south west corner and north east corner.
template<int d, typename T>
void
TensorBsplineBasis<d, T>::KnotSpanGetter(TensorBsplineBasis<d, T>::KnotSpanList& knot_spans) const
{
    knot_spans.clear();
    std::array<std::vector<std::pair<T, T>>, d> knot_span_in;
    for (int i = 0; i<d; ++i)
    {
        knot_span_in[i] = KnotVectorGetter(i).KnotSpans();
    }
    std::vector<int> indexes(d, 0);
    std::vector<int> endIndex(d);
    for (int i = 0; i!=d; ++i)
    {
        endIndex[i] = knot_span_in[i].size();
    }
    std::function<void(std::vector<int>&, int)> recursive;
    vector start_knot(d), end_knot(d);
    recursive =
            [this, &endIndex, &knot_spans, &knot_span_in, &start_knot, &end_knot, &recursive](std::vector<int>& indexes,
                    int direction)
            {
                if (direction==d)
                {
                    knot_spans.push_back(std::make_pair(start_knot, end_knot));
                }
                else
                {
                    for (indexes[direction] = 0; indexes[direction]!=endIndex[direction]; indexes[direction]++)
                    {
                        start_knot(direction) = knot_span_in[direction][indexes[direction]].first;
                        end_knot(direction) = knot_span_in[direction][indexes[direction]].second;
                        recursive(indexes, direction+1);
                    }
                }
            };
    recursive(indexes, 0);
}

template<int d, typename T>
bool TensorBsplineBasis<d, T>::InDomain(const TensorBsplineBasis<d, T>::vector& u) const
{
    ASSERT((u.size()==d), "Invalid input vector size.");
    for (int i = 0; i<d; ++i)
    {
        if (u(i)<DomainStart(i) || u(i)>DomainEnd(i))
        {
            return false;
        }
    }
    return true;
}

template<typename T>
class TensorBsplineBasis<0, T> {
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

    TensorBsplineBasis() { };

    TensorBsplineBasis(const T& support)
            :_basis{support} { };

    ~TensorBsplineBasis() { };

    BasisFunValDerAllList_ptr
    EvalDualAllTensor(const vector& u) const
    {
        BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
        std::vector<T> result(1, 0);
        if (u(0)==_basis)
        {
            result[0] = 1;
        }
        Result->push_back(BasisFunValDerAll(0, result));
        return Result;
    }

    BasisFunValDerAllList_ptr
    EvalDerAllTensor(const vector& u, const int i = 0) const
    {
        BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
        std::vector<T> result(i+1, 0);
        if (u(0)==_basis)
        {
            for (int j = 0; j<result.size(); ++j)
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
