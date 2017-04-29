//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"
#include "MultiArray.h"
#include <algorithm>

namespace Accessory {
    using namespace Eigen;
    template<typename T, unsigned N>
    using ContPtrList = std::vector<Matrix<T, N, 1>>;

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

    template<typename T, unsigned N>
    void degreeElevate(int t, KnotVector<T> &U, ContPtrList<T, N> &P) {
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

    template<typename T, unsigned N>
    void knotInsertion(T u, int r, KnotVector<T> &U, ContPtrList<T, N> &P) {

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

        ContPtrList<T, N> R(p + 1);
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

    template<typename T, unsigned N>
    void refineKnotVectorCurve(const KnotVector<T> &X, KnotVector<T> &U, ContPtrList<T, N> &P) {

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
}

template<unsigned d, unsigned N, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using Ptr=Eigen::Matrix<T, N, 1>;
    using GeometryVector = std::vector<Ptr>;
    typedef std::vector<unsigned> DiffPattern;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const BsplineBasis<T> &,
                          const GeometryVector &);

    Ptr AffineMap(const Ptr &, const DiffPattern &i = DiffPattern(d, 0)) const;

    T Jacobian(const Ptr &) const;

    void DegreeElevate(unsigned, unsigned);

    void UniformRefine(unsigned, unsigned, unsigned m = 1);

    void PrintCtrPtr() const;

    virtual ~PhyTensorBsplineBasis() {

    }

private:
    GeometryVector _geometricInfo;
};

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis():TensorBsplineBasis<d, T>() {

}

template<unsigned d, unsigned N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::Ptr
PhyTensorBsplineBasis<d, N, T>::AffineMap(const PhyTensorBsplineBasis<d, N, T>::Ptr &u, const DiffPattern &i) const {
    PhyTensorBsplineBasis<d, N, T>::Ptr result;
    result.setZero();
    auto p = this->TensorBsplineBasis<d, T>::EvalTensor(u,i);
    for (auto it = p->begin(); it != p->end(); ++it) {
        result += _geometricInfo[it->first] * it->second;
    }
    return result;
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX,
                                                      const PhyTensorBsplineBasis<d, N, T>::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const BsplineBasis<T> &baseZ,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY, baseZ), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::DegreeElevate(unsigned orientation, unsigned r) {
    std::vector<unsigned> indexes(d, 0);
    std::vector<unsigned> endPerIndex;
    for (unsigned direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    GeometryVector temp1;
    std::vector<unsigned> MultiIndex(d);
    std::function<void(std::vector<unsigned> &, const std::vector<unsigned> &, unsigned)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, r, &temp1, &MultiIndex, &recursive](
            std::vector<unsigned> &indexes,
            const std::vector<unsigned> &endPerIndex,
            unsigned direction) {
        if (direction == indexes.size()) {

            Accessory::ContPtrList<T, N> ElevateList;
            for (unsigned i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                ElevateList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::degreeElevate<T, N>(r, tmp, ElevateList);
            if (knot_temp_storage.GetSize() == 0) {
                for (unsigned i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (unsigned i = 0; i != ElevateList.size(); ++i) {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index(MultiIndex);
                temp1[index] = ElevateList[i];
            }
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                MultiIndex[direction] = indexes[direction];
                if (direction == orientation && called == false) {
                    called = true;
                } else if (direction == orientation && called == true) {
                    break;
                }
                called = false;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefine(unsigned orientation, unsigned r, unsigned m) {
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    std::vector<unsigned> indexes(d, 0);
    std::vector<unsigned> endPerIndex;
    for (unsigned direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    GeometryVector temp1;
    std::vector<unsigned> MultiIndex(d);
    std::function<void(std::vector<unsigned> &, const std::vector<unsigned> &, unsigned)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage(this->_basis[orientation].Knots());
    knot_temp_storage.UniformRefine(r, m);
    KnotVector<T> X = knot_temp_storage.Difference(this->_basis[orientation].Knots());
    knot_temp_storage.resize(0);
    bool called = false;
    recursive = [this, &X, &orientation, &called, &knot_temp_storage, &tmp1, &temp1, &MultiIndex, &recursive](
            std::vector<unsigned> &indexes, const std::vector<unsigned> &endPerIndex, unsigned direction) {
        if (direction == indexes.size()) {
            Accessory::ContPtrList<T, N> RefineList;
            for (unsigned i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                RefineList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::refineKnotVectorCurve<T, N>(X, tmp, RefineList);
            if (knot_temp_storage.GetSize() == 0) {
                for (unsigned i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (unsigned i = 0; i != RefineList.size(); ++i) {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index(MultiIndex);
                temp1[index] = RefineList[i];
            }
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                MultiIndex[direction] = indexes[direction];
                if (direction == orientation && called == false) {
                    called = true;
                } else if (direction == orientation && called == true) {
                    break;
                }
                called = false;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::PrintCtrPtr() const {
    for (int i = 0; i != _geometricInfo.size(); ++i)
        std::cout << _geometricInfo[i].transpose() << std::endl;
}

template<unsigned d, unsigned N, typename T>
T PhyTensorBsplineBasis<d, N, T>::Jacobian(const PhyTensorBsplineBasis::Ptr &u) const {
    ASSERT(d == N, "Dimension should be the same between parametric space and Physical space.");
    using JacobianMatrix=Eigen::Matrix<T, d, d>;
    JacobianMatrix a;
    for (int i = 0; i != d; i++) {
        std::vector<unsigned> differentiation(d, 0);
        differentiation[i] = 1;
        auto aa = AffineMap(u,differentiation);
        a.col(i)=aa;
    }
    return a.determinant();
};
#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
