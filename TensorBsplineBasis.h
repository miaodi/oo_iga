//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"
#include <array>
#include <algorithm>

namespace Accessory {
    using namespace Eigen;
    template<typename T, int N>
    using ContPtsList = std::vector<Matrix<T, N, 1>>;

    using DifferentialPattern = std::vector<int>;
    using DifferentialPatternList = std::vector<DifferentialPattern>;
    using DifferentialPatternList_ptr = std::unique_ptr<DifferentialPatternList>;

    template<typename T>
    void binomialCoef(Matrix<T, Dynamic, Dynamic> &Bin) {
        int n, k;
        // Setup the first line
        Bin(0, 0) = 1.0;
        for (k = static_cast<int>(Bin.cols()) - 1; k > 0; --k)
            Bin(0, k) = 0.0;
        // Setup the other lines
        for (n = 0; n < static_cast<int>(Bin.rows()) - 1; n++) {
            Bin(n + 1, 0) = 1.0;
            for (k = 1; k < static_cast<int>(Bin.cols()); k++)
                if (n + 1 < k)
                    Bin(n, k) = 0.0;
                else
                    Bin(n + 1, k) = Bin(n, k) + Bin(n, k - 1);
        }
    }

    template<typename T, int N>
    void degreeElevate(int t, KnotVector<T> &U, ContPtsList<T, N> &P) {
        ASSERT(t > 0, "Invalid geometrical information input, check size bro.");
        int i, j, k;
        auto dof = U.GetDOF();
        auto cP = P;
        auto cU = U;
        int n = dof - 1;
        int p = U.GetDegree();
        int m = n + p + 1;
        int ph = p + t;
        int ph2 = ph / 2;
        Matrix<T, Dynamic, Dynamic> bezalfs(p + t + 1, p + 1); // coefficients for degree elevating the Bezier segment
        Matrix<Matrix<T, N, 1>, Dynamic, 1> bpts(p + 1); // pth-degree Bezier control points of the current segment
        Matrix<Matrix<T, N, 1>, Dynamic, 1> ebpts(
                p + t + 1); // (p+t)th-degree Bezier control points of the  current segment
        Matrix<Matrix<T, N, 1>, Dynamic, 1> Nextbpts(p - 1); // leftmost control points of the next Bezier segment
        Matrix<T, Dynamic, 1> alphas(p - 1); // knot instertion alphas.
        // Compute the binomial coefficients
        Matrix<T, Dynamic, Dynamic> Bin(ph + 1, ph2 + 1);
        bezalfs.setZero();
        alphas.setZero();
        Bin.setZero();
        binomialCoef(Bin);

        // Compute Bezier degree elevation coefficients
        T inv, mpi;
        bezalfs(0, 0) = bezalfs(ph, p) = 1.0;
        for (i = 1; i <= ph2; i++) {
            inv = 1.0 / Bin(ph, i);
            mpi = std::min(p, i);
            for (j = std::max(0, i - t); j <= mpi; j++) {
                bezalfs(i, j) = inv * Bin(p, j) * Bin(t, i - j);
            }
        }

        for (i = ph2 + 1; i < ph; i++) {
            mpi = std::min(p, i);
            for (j = std::max(0, i - t); j <= mpi; j++)
                bezalfs(i, j) = bezalfs(ph - i, p - j);
        }

        P.resize(cP.size() * t * 3); // Allocate more control points than necessary
        U.resize(cP.size() * t * 3 + ph + 1);
        int mh = ph;
        int kind = ph + 1;
        T ua = U(0);
        T ub = 0.0;
        int r = -1;
        int oldr;
        int a = p;
        int b = p + 1;
        int cind = 1;
        int rbz, lbz = 1;
        int mul, save, s;
        T alf;
        int first, last, kj;
        T den, bet, gam, numer;

        P[0] = cP[0];
        for (i = 0; i <= ph; i++) {
            U(i) = ua;
        }

        // Initialize the first Bezier segment
        for (i = 0; i <= p; i++)
            bpts(i) = cP[i];
        while (b < m) { // Big loop thru knot vector
            i = b;
            while (b < m && cU(b) == cU(b + 1)) // for some odd reasons... == doesn't work
                b++;
            mul = b - i + 1;
            mh += mul + t;
            ub = cU(b);
            oldr = r;
            r = p - mul;
            if (oldr > 0)
                lbz = (oldr + 2) / 2;
            else
                lbz = 1;
            if (r > 0)
                rbz = ph - (r + 1) / 2;
            else
                rbz = ph;
            if (r > 0) { // Insert knot to get Bezier segment
                numer = ub - ua;
                for (k = p; k > mul; k--) {
                    alphas(k - mul - 1) = numer / (cU(a + k) - ua);
                }
                for (j = 1; j <= r; j++) {
                    save = r - j;
                    s = mul + j;
                    for (k = p; k >= s; k--) {
                        bpts(k) = alphas(k - s) * bpts(k) + (1.0 - alphas(k - s)) * bpts(k - 1);
                    }
                    Nextbpts(save) = bpts(p);
                }
            }

            for (i = lbz; i <= ph; i++) { // Degree elevate Bezier,  only the points lbz,...,ph are used
                ebpts(i) = Matrix<T, Dynamic, 1>::Zero(N);
                mpi = std::min(p, i);
                for (j = std::max(0, i - t); j <= mpi; j++)
                    ebpts(i) += bezalfs(i, j) * bpts(j);
            }

            if (oldr > 1) { // Must remove knot u=c.U[a] oldr times
                // if(oldr>2) // Alphas on the right do not change
                //	alfj = (ua-U[kind-1])/(ub-U[kind-1]) ;
                first = kind - 2;
                last = kind;
                den = ub - ua;
                bet = (ub - U(kind - 1)) / den;
                for (int tr = 1; tr < oldr; tr++) { // Knot removal loop
                    i = first;
                    j = last;
                    kj = j - kind + 1;
                    while (j - i > tr) { // Loop and compute the new control points for one removal step
                        if (i < cind) {
                            alf = (ub - U(i)) / (ua - U(i));
                            P[i] = alf * P[i] + (1.0 - alf) * P[i - 1];
                        }
                        if (j >= lbz) {
                            if (j - tr <= kind - ph + oldr) {
                                gam = (ub - U(j - tr)) / den;
                                ebpts(kj) = gam * ebpts(kj) + (1.0 - gam) * ebpts(kj + 1);
                            } else {
                                ebpts(kj) = bet * ebpts(kj) + (1.0 - bet) * ebpts(kj + 1);
                            }
                        }
                        ++i;
                        --j;
                        --kj;
                    }
                    --first;
                    ++last;
                }
            }

            if (a != p) // load the knot u=c.U[a]
                for (i = 0; i < ph - oldr; i++) {
                    U(kind++) = ua;
                }
            for (j = lbz; j <= rbz; j++) { // load control points onto the curve
                P[cind++] = ebpts(j);
            }

            if (b < m) { // Set up for next pass thru loop
                for (j = 0; j < r; j++)
                    bpts(j) = Nextbpts(j);
                for (j = r; j <= p; j++)
                    bpts(j) = cP[b - p + j];
                a = b;
                b++;
                ua = ub;
            } else {
                for (i = 0; i <= ph; i++)
                    U(kind + i) = ub;
            }
        }
        P.resize(mh - ph); // Resize to the proper number of control points
        U.resize(mh + 1);
    }

    template<typename T, int N>
    void knotInsertion(T u, int r, KnotVector<T> &U, ContPtsList<T, N> &P) {

        int n = U.GetDOF();
        int p = U.GetDegree();
        auto cP = P;
        auto cU = U;
        int m = n + p;
        int nq = n + r;
        int k, s = 0;
        int i, j;
        k = U.FindSpan(u);
        P.resize(nq);
        U.resize(nq + p + 1);
        for (i = 0; i <= k; i++)
            U(i) = cU(i);
        for (i = 1; i <= r; i++)
            U(k + i) = u;
        for (i = k + 1; i <= m; i++)
            U(i + r) = cU(i);

        ContPtsList<T, N> R(p + 1);
        for (i = 0; i <= k - p; i++)
            P[i] = cP[i];
        for (i = k - s; i < n; i++)
            P[i + r] = cP[i];
        for (i = 0; i <= p - s; i++)
            R[i] = cP[k - p + i];
        int L;
        T alpha;
        for (j = 1; j <= r; j++) {
            L = k - p + j;
            for (i = 0; i <= p - j - s; i++) {
                alpha = (u - cU(L + i)) / (cU(i + k + 1) - cU(L + i));
                R[i] = alpha * R[i + 1] + (1.0 - alpha) * R(i);
            }
            P[L] = R[0];
            P[k + r - j - s] = R[p - j - s];
        }
        for (i = L + 1; i < k - s; i++)
            P[i] = R[i - L];

    }

    template<typename T, int N>
    void refineKnotVectorCurve(const KnotVector<T> &X, KnotVector<T> &U, ContPtsList<T, N> &P) {

        int n = U.GetDOF() - 1;
        int p = U.GetDegree();
        int m = n + p + 1;
        int a, b;
        int r = static_cast<int>(X.GetSize() - 1);
        auto cP = P;
        auto cU = U;
        P.resize(r + n + 2);
        U.resize(r + n + p + 3);
        a = cU.FindSpan(X[0]);
        b = cU.FindSpan(X[r]);
        ++b;
        int j;
        for (j = 0; j <= a - p; j++)
            P[j] = cP[j];
        for (j = b - 1; j <= n; j++)
            P[j + r + 1] = cP[j];
        for (j = 0; j <= a; j++)
            U(j) = cU(j);
        for (j = b + p; j <= m; j++)
            U(j + r + 1) = cU(j);
        int i = b + p - 1;
        int k = b + p + r;
        for (j = r; j >= 0; j--) {
            while (X[j] <= cU[i] && i > a) {
                P[k - p - 1] = cP[i - p - 1];
                U(k) = cU(i);
                --k;
                --i;
            }
            P[k - p - 1] = P[k - p];
            for (int l = 1; l <= p; l++) {
                int ind = k - p + l;
                T alpha = U[k + l] - X[j];
                if (alpha == 0.0)
                    P[ind - 1] = P[ind];
                else
                    alpha /= U(k + l) - cU(i - p + l);
                P[ind - 1] = alpha * P[ind - 1] + (1.0 - alpha) * P[ind];
            }
            U(k) = X[j];
            --k;
        }
    }

    template<int N>
    DifferentialPatternList_ptr PartialDerPattern(int r) {
        std::vector<int> kk(r);
        DifferentialPatternList_ptr a(new DifferentialPatternList);
        std::function<void(int, int, std::vector<int> &, int, int,
                           DifferentialPatternList_ptr &)> recursive;
        recursive = [&](int D, int i, std::vector<int> &k, int n, int start,
                        std::unique_ptr<std::vector<std::vector<int>>> &a) {
            if (n == i) {
                std::vector<int> m;
                int it = 0;
                for (int it1 = 0; it1 < D; ++it1) {
                    int amount = 0;
                    while (find(k.begin(), k.end(), it) != k.end()) {
                        amount++;
                        it++;
                    }
                    m.push_back(amount);
                    it++;
                }
                a->push_back(m);
            } else {
                for (int jj = start; jj < D + i - (i - n); ++jj) {
                    k[n] = jj;
                    recursive(D, i, k, n + 1, jj + 1, a);
                }
            }
        };
        recursive(N, r, kk, 0, 0, a);
        return a;
    }
}

template<int d, typename T=double>
class TensorBsplineBasis {
public:

    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef std::vector<int> DiffPattern;

    typedef typename BsplineBasis<T>::BasisFunVal BasisFunVal;
    typedef typename BsplineBasis<T>::BasisFunValPac BasisFunValPac;
    typedef typename BsplineBasis<T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    typedef typename BsplineBasis<T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename BsplineBasis<T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;

    TensorBsplineBasis();

    TensorBsplineBasis(const BsplineBasis<T> &baseX);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY);

    TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY, const BsplineBasis<T> &baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>> &KV);

    virtual ~TensorBsplineBasis() {};

    int GetDegree(const int i) const;

    int GetDof() const;

    int GetDof(const int i) const;

    std::vector<int> TensorIndex(const int &m) const {
        ASSERT(m < GetDof(), "Input index is invalid.");
        std::vector<int> ind(d);
        int mm = m;
        ///int always >=0.
        for (int i = static_cast<int>(d - 1); i >= 0; --i) {
            ind[i] = mm % GetDof(i);
            mm -= ind[i];
            mm /= GetDof(i);
        }
        return ind;
    }

    int Index(const std::vector<int> &ts) const {
        ASSERT(ts.size() == d, "Input index is invalid.");
        int index = 0;
        for (int direction = 0; direction < d; ++direction) {
            index += ts[direction];
            if (direction != d - 1) { index *= GetDof(direction + 1); }
        }
        return index;
    }

    std::unique_ptr<std::vector<int>> AllActivatedDofsOnBoundary(const int &, const int &) const;

    matrix Support(const int &i) const {
        matrix res(static_cast<int>(d), 2);
        auto ti = TensorIndex(i);
        for (int j = 0; j != d; ++j)
            res.row(j) = _basis[j].Support(ti[j]);
        return res;
    }

    int NumActive() const;

    int NumActive(const int &i) const;

    void ChangeKnots(const KnotVector<T> &, int);

    void PrintKnots(int i) const { _basis[i].PrintKnots(); }

    void PrintUniKnots(int i) const { _basis[i].PrintUniKnots(); }

    const KnotVector<T> &KnotVectorGetter(int i) const {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].Knots();
    }

    const T DomainStart(int i) const {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].DomainStart();
    }

    const T DomainEnd(int i) const {
        ASSERT(i < d, "Invalid dimension index provided.");
        return _basis[i].DomainEnd();
    }

    BasisFunValPac_ptr EvalTensor(const vector &u, const DiffPattern &i = DiffPattern(d, 0)) const;

    BasisFunValDerAllList_ptr EvalDerAllTensor(const vector &u, const int i = 0) const;

    std::vector<int> ActiveIndex(const vector &u) const {
        std::vector<int> temp;
        temp.reserve(NumActive());
        ASSERT((u.size() == d), "Invalid input vector size.");
        std::vector<int> indexes(d, 0);
        std::vector<int> endPerIndex(d);
        std::vector<int> startIndex(d);
        for (int i = 0; i != d; ++i) {
            startIndex[i] = _basis[i].FirstActive(u(i));
            endPerIndex[i] = _basis[i].NumActive();
        }
        std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
        std::vector<int> multiIndex(d);
        recursive = [this, &startIndex, &temp, &multiIndex, &recursive](
                std::vector<int> &indexes,
                const std::vector<int> &endPerIndex,
                int direction) {
            if (direction == indexes.size()) {
                temp.push_back(Index(multiIndex));

            } else {
                for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                    multiIndex[direction] = startIndex[direction] + indexes[direction];
                    recursive(indexes, endPerIndex, direction + 1);
                }
            }
        };
        recursive(indexes, endPerIndex, 0);
        return temp;
    }

    T EvalSingle(const vector &u, const int n, const TensorBsplineBasis::DiffPattern &i) {
        ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
        auto tensorindex = TensorIndex(n);
        T result = 1;
        for (int direction = 0; direction != d; ++direction) {
            result *= _basis[direction].EvalSingle(u(direction), tensorindex[direction], i[direction]);
        }
        return result;
    }


protected:
    std::array<BsplineBasis<T>,d> _basis;

    TensorBsplineBasis(const TensorBsplineBasis<d, T> &) {};

    TensorBsplineBasis operator=(const TensorBsplineBasis<d, T> &) {};

};

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis() {
    for (int i = 0; i < d; ++i)
        _basis[i] = BsplineBasis<T>();
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX) {
    ASSERT(d == 1, "Invalid dimension.");
    _basis[0] = baseX;

}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY) {
    ASSERT(d == 2, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                             const BsplineBasis<T> &baseZ) {
    ASSERT(d == 3, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
}

template<int d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const std::vector<KnotVector<T>> &knotVectors) {
    ASSERT(d == knotVectors.size(), "Invalid number of knot-vectors given.");
    for (int i = 0; i != d; ++i)
        _basis[i] = BsplineBasis<T>(knotVectors[i]);
}

template<int d, typename T>
int TensorBsplineBasis<d, T>::GetDegree(const int i) const {
    return _basis[i].GetDegree();
}

template<int d, typename T>
int TensorBsplineBasis<d, T>::GetDof() const {
    int dof = 1;
    for (int i = 0; i != d; ++i)
        dof *= _basis[i].GetDof();
    return dof;
}

template<int d, typename T>
int TensorBsplineBasis<d, T>::GetDof(const int i) const {
    return _basis[i].GetDof();
}

template<int d, typename T>
int TensorBsplineBasis<d, T>::NumActive(const int &i) const {
    return _basis[i].NumActive();
}

template<int d, typename T>
int TensorBsplineBasis<d, T>::NumActive() const {
    int active = 1;
    for (int i = 0; i != d; ++i)
        active *= _basis[i].NumActive();
    return active;
}

template<int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr
TensorBsplineBasis<d, T>::EvalTensor(const TensorBsplineBasis::vector &u,
                                     const TensorBsplineBasis::DiffPattern &i) const {
    ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    std::array<BasisFunValPac_ptr, d> OneDResult;
    for (int direction = 0; direction != d; ++direction) {
        OneDResult[direction] = _basis[direction].Eval(u(direction), i[direction]);
        endPerIndex.push_back(OneDResult[direction]->size());
    }
    std::vector<int> MultiIndex(d);
    std::vector<T> Value(d);
    BasisFunValPac_ptr Result(new BasisFunValPac);

    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;

    recursive = [this, &OneDResult, &MultiIndex, &Value, &Result, &recursive](std::vector<int> &indexes,
                                                                              const std::vector<int> &endPerIndex,
                                                                              int direction) {
        if (direction == indexes.size()) {
            T result = 1;
            for (int ii = 0; ii < d; ii++)
                result *= Value[ii];
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

template<int d, typename T>
void TensorBsplineBasis<d, T>::ChangeKnots(const KnotVector<T> &knots, int direction) {
    _basis[direction] = knots;
}

template<int d, typename T>
typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr
TensorBsplineBasis<d, T>::EvalDerAllTensor(const TensorBsplineBasis::vector &u,
                                           const int i) const {
    ASSERT((u.size() == d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    Accessory::DifferentialPatternList differentialPatternList;
    for (int order = 0; order <= i; ++order) {
        auto temp = Accessory::PartialDerPattern<d>(order);
        differentialPatternList.insert(differentialPatternList.end(), temp->begin(), temp->end());
    }
    int derivativeAmount = differentialPatternList.size();
    auto numActived = NumActive();
    BasisFunValDerAllList_ptr Result(new BasisFunValDerAllList);
    std::array<BasisFunValDerAllList_ptr, d> oneDResult;
    for (int direction = 0; direction != d; ++direction) {
        oneDResult[direction] = _basis[direction].EvalDerAll(u(direction), i);
        endPerIndex.push_back(oneDResult[direction]->size());
    }

    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;

    std::vector<int> multiIndex(d);
    std::vector<std::vector<T>> Values(derivativeAmount, std::vector<T>(d, 0));
    recursive = [this, &derivativeAmount, &oneDResult, &multiIndex, &Values, &Result, &differentialPatternList, &recursive](
            std::vector<int> &indexes,
            const std::vector<int> &endPerIndex,
            int direction) {
        if (direction == indexes.size()) {
            std::vector<T> result(derivativeAmount, 1);
            for (int iii = 0; iii != derivativeAmount; ++iii) {
                for (int ii = 0; ii != d; ++ii) {
                    result[iii] *= Values[iii][ii];
                }
            }
            Result->push_back(BasisFunValDerAll(Index(multiIndex), result));
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                for (auto it_diffPart = differentialPatternList.begin();
                     it_diffPart != differentialPatternList.end(); ++it_diffPart) {
                    int diffPart_label = it_diffPart - differentialPatternList.begin();
                    Values[diffPart_label][direction] = (*oneDResult[direction])[indexes[direction]].second[(*it_diffPart)[direction]];
                }
                multiIndex[direction] = (*oneDResult[direction])[indexes[direction]].first;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    return Result;
}

template<int d, typename T>
std::unique_ptr<std::vector<int>>
TensorBsplineBasis<d, T>::AllActivatedDofsOnBoundary(const int &orientation, const int &layer) const {
    ASSERT(orientation < d, "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex(d, 0);
    for (int i = 0; i != d; ++i) {
        if (i == orientation) {
            endPerIndex[i] = 1;
        } else {
            endPerIndex[i] = GetDof(i);
        }
    }
    std::unique_ptr<std::vector<int>> result(new std::vector<int>);
    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
    std::vector<int> temp(d, 0);
    recursive = [this, &orientation, &layer, &result, &temp, &recursive](
            std::vector<int> &indexes,
            const std::vector<int> &endPerIndex,
            int direction) {
        if (direction == d) {
            result->push_back(Index(temp));
        } else {
            if (direction == orientation) {
                temp[direction] = layer;
                recursive(indexes, endPerIndex, direction + 1);
            } else {
                for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                    temp[direction] = indexes[direction];
                    recursive(indexes, endPerIndex, direction + 1);
                }
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    return result;
}




#endif //OO_IGA_TENSORBSPLINEBASIS_H
