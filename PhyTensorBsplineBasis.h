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
    using ContPtrList = Matrix<Matrix<T, N, 1>, Dynamic, 1>;

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
    void degreeElevate(int t, KnotVector<T> U, ContPtrList<T,N>& P) {
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
        Matrix<Matrix<T, N, 1>, Dynamic, 1> ebpts(p + t + 1); // (p+t)th-degree Bezier control points of the  current segment
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

        P.conservativeResize(cP.rows() * t * 3); // Allocate more control points than necessary

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

        P(0) = cP(0);
        for (i = 0; i <= ph; i++) {
            U(i) = ua;
        }

        // Initialize the first Bezier segment
        for (i = 0; i <= p; i++)
            bpts(i) = cP(i);
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
                            P(i) = alf * P(i) + (1.0 - alf) * P(i - 1);
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
                P(cind++) = ebpts(j);
            }

            if (b < m) { // Set up for next pass thru loop
                for (j = 0; j < r; j++)
                    bpts(j) = Nextbpts(j);
                for (j = r; j <= p; j++)
                    bpts(j) = cP(b - p + j);
                a = b;
                b++;
                ua = ub;
            } else {
                for (i = 0; i <= ph; i++)
                    U(kind + i) = ub;
            }
        }
        P.conservativeResize(mh - ph); // Resize to the proper number of control points

    }

}
template<unsigned d, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using EigenVector=Eigen::Matrix<T, Eigen::Dynamic, 1>;

    using GeometryVector = std::vector<EigenVector>;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, GeometryVector geometry);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &KV, GeometryVector geometry);

    EigenVector AffineMap(const EigenVector &u) const;

    virtual ~PhyTensorBsplineBasis() {

    }

private:
    GeometryVector _geometricInfo;
};

template<unsigned d, typename T>
PhyTensorBsplineBasis<d, T>::PhyTensorBsplineBasis():TensorBsplineBasis<d, T>() {

}

template<unsigned d, typename T>
PhyTensorBsplineBasis<d, T>::PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &KV, GeometryVector geometry) : TensorBsplineBasis<d, T>(KV) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(), "Invalid geometrical information input, check size bro.");
    _geometricInfo = geometry;
}

template<unsigned d, typename T>
typename PhyTensorBsplineBasis<d, T>::EigenVector PhyTensorBsplineBasis<d, T>::AffineMap(const PhyTensorBsplineBasis<d, T>::EigenVector &u) const {
    PhyTensorBsplineBasis<d, T>::EigenVector result(d);
    result.setZero();
    auto p = this->TensorBsplineBasis<d, T>::EvalTensor(u);
    for (auto it = p->begin(); it != p->end(); ++it) {
        result += _geometricInfo[it->first] * (it->second)[0][0];
    }
    return result;
}


#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
