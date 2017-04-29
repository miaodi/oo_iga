//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"
#include <array>

template<unsigned d, typename T=double>
class TensorBsplineBasis {
public:

    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef std::vector<unsigned> DiffPattern;

    typedef typename BsplineBasis<T>::BasisFunVal BasisFunVal;
    typedef typename BsplineBasis<T>::BasisFunValPac BasisFunValPac;
    typedef typename BsplineBasis<T>::BasisFunValPac_ptr BasisFunValPac_ptr;

    TensorBsplineBasis();

    TensorBsplineBasis(const BsplineBasis<T> &baseX);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY, const BsplineBasis<T> &baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>> &KV);

    virtual ~TensorBsplineBasis() {};

    unsigned GetDegree(const unsigned i) const;

    unsigned GetDof() const;

    unsigned GetDof(const unsigned i) const;

    std::vector<unsigned> TensorIndex(const unsigned &m) const {
        ASSERT(m < GetDof(), "Input index is invalid.");
        std::vector<unsigned> ind(d);
        int mm = m;
        ///unsigned always >=0.
        for (int i = static_cast<int>(d - 1); i >= 0; --i) {
            ind[i] = mm % GetDof(i);
            mm -= ind[i];
            mm /= GetDof(i);
        }
        return ind;
    }

    unsigned Index(const std::vector<unsigned> &ts) const {
        ASSERT(ts.size() == d, "Input index is invalid.");
        unsigned index = 0;
        for (unsigned direction = 0; direction < d; ++direction) {
            index += ts[direction];
            if (direction != d - 1) { index *= GetDof(direction + 1); }
        }
        return index;
    }

    matrix Support(const unsigned &i) const {
        matrix res(static_cast<int>(d), 2);
        auto ti = TensorIndex(i);
        for (unsigned j = 0; j != d; ++j)
            res.row(j) = _basis[j].Support(ti[j]);
        return res;
    }

    unsigned NumActive() const;

    unsigned NumActive(const unsigned &i) const;

    void ChangeKnots(const KnotVector<T> &, unsigned);

    void PrintKnots(unsigned i) const { _basis[i].PrintKnots(); }

    void PrintUniKnots(unsigned i) const { _basis[i].PrintUniKnots(); }

    BasisFunValPac_ptr EvalTensor(const vector &u, const DiffPattern &i = DiffPattern(d, 0)) const;

    T EvalSingle(const vector &u, const unsigned n, const TensorBsplineBasis::DiffPattern &i) {
        ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
        auto tensorindex = TensorIndex(n);
        T result = 1;
        for (unsigned direction = 0; direction != d; ++direction) {
            result *= _basis[direction].EvalSingle(u(direction), tensorindex[direction], i[direction]);
        }
        return result;
    }


protected:
    BsplineBasis<T> _basis[d];

    TensorBsplineBasis(const TensorBsplineBasis<d, T> &) {};

    TensorBsplineBasis operator=(const TensorBsplineBasis<d, T> &) {};

};

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis() {
    for (unsigned i = 0; i < d; ++i)
        _basis[i] = BsplineBasis<T>();
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX) {
    ASSERT(d == 1, "Invalid dimension.");
    _basis[0] = baseX;

}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY) {
    ASSERT(d == 2, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                             const BsplineBasis<T> &baseZ) {
    ASSERT(d == 3, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const std::vector<KnotVector<T>> &knotVectors) {
    ASSERT(d == knotVectors.size(), "Invalid number of knot-vectors given.");
    for (unsigned i = 0; i != d; ++i)
        _basis[i] = BsplineBasis<T>(knotVectors[i]);
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDegree(const unsigned i) const {
    return _basis[i].GetDegree();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDof() const {
    unsigned dof = 1;
    for (unsigned i = 0; i != d; ++i)
        dof *= _basis[i].GetDof();
    return dof;
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDof(const unsigned i) const {
    return _basis[i].GetDof();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::NumActive(const unsigned &i) const {
    return _basis[i].NumActive();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::NumActive() const {
    unsigned active = 1;
    for (unsigned i = 0; i != d; ++i)
        active *= _basis[i].NumActive();
    return active;
}

template<unsigned d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr
TensorBsplineBasis<d, T>::EvalTensor(const TensorBsplineBasis::vector &u,
                                     const TensorBsplineBasis::DiffPattern &i) const {

    ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
    std::vector<unsigned> indexes(d, 0);
    std::vector<unsigned> endPerIndex;
    std::array<BasisFunValPac_ptr, d> OneDResult;
    for (unsigned direction = 0; direction != d; ++direction) {
        OneDResult[direction] = _basis[direction].Eval(u(direction), i[direction]);
        endPerIndex.push_back(OneDResult[direction]->size());
    }
    std::vector<unsigned> MultiIndex(d);
    std::vector<T> Value(d);
    BasisFunValPac_ptr Result(new BasisFunValPac);

    std::function<void(std::vector<unsigned> &, const std::vector<unsigned> &, unsigned)> recursive;

    recursive = [this, &OneDResult, &MultiIndex, &Value, &Result, &recursive](std::vector<unsigned> &indexes,
                                                                              const std::vector<unsigned> &endPerIndex,
                                                                              unsigned direction) {
        if (direction == indexes.size()) {
            T result = 1;
            for (unsigned i = 0; i < d; i++)
                result *= Value[i];
            Result->push_back(BasisFunVal(Index(MultiIndex), result));
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                Value[direction] = (*OneDResult[direction])[indexes[direction]].second;
                MultiIndex[direction] = (*OneDResult[direction])[indexes[direction]].first;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);

    return Result;
}

template<unsigned d, typename T>
void TensorBsplineBasis<d, T>::ChangeKnots(const KnotVector<T> &knots, unsigned direction) {
    _basis[direction] = knots;
}


#endif //OO_IGA_TENSORBSPLINEBASIS_H
