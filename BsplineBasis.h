//
// Created by miaodi on 25/12/2016.
//

#ifndef OO_IGA_BSPLINEBASIS_H
#define OO_IGA_BSPLINEBASIS_H
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#include "KnotVector.h"
#include <memory>
#include <map>
#include <eigen3/Eigen/Dense>

template<typename T=double>
class BsplineBasis {
public:
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using block = Eigen::Block<matrix>;

    typedef std::pair<unsigned, T> BasisFunVal;
    typedef std::vector<BasisFunVal> BasisFunValPac;
    typedef std::unique_ptr<BasisFunValPac> BasisFunValPac_ptr;
    template<unsigned N>
    using BasisFunValDerAll=std::pair<unsigned, std::array<T, N>>;
    template<unsigned N, unsigned Degree>
    using BasisFunValDerAllList =std::array<BasisFunValDerAll<N>, Degree>;
    template<unsigned N, unsigned Degree>
    using BasisFunValDerAllList_ptr=std::unique_ptr<BasisFunValDerAllList<N, Degree>>;

    BsplineBasis();

    BsplineBasis(KnotVector<T> target);

    unsigned GetDegree() const;

    unsigned GetDof() const;

    virtual ~BsplineBasis();

    unsigned FindSpan(const T &u) const;

    template<unsigned Derivative, unsigned Degree>
    BasisFunValDerAllList_ptr<Derivative + 1, Degree + 1> EvalDerAll(const T &u) const {
        auto i = Degree+1;
        const unsigned dof = GetDof();
        const unsigned deg = GetDegree();
        BasisFunValDerAllList_ptr<Derivative + 1,Degree+1> ders(new BasisFunValDerAllList<Derivative + 1, Degree + 1>);

        T *left = new T[2 * (deg + 1)];
        T *right = &left[deg + 1];
        matrix ndu(deg + 1, deg + 1);
        T saved, temp;
        int j, r;
        unsigned span = FindSpan(u);
        ndu(0, 0) = 1.0;
        for (j = 1; j <= deg; j++) {
            left[j] = u - _basisKnot[span + 1 - j];
            right[j] = _basisKnot[span + j] - u;
            saved = 0.0;

            for (r = 0; r < j; r++) {
                // Lower triangle
                ndu(j, r) = right[r + 1] + left[j - r];
                temp = ndu(r, j - 1) / ndu(j, r);
                // _basisKnotpper triangle
                ndu(r, j) = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            ndu(j, j) = saved;
        }

        for (j = deg; j >= 0; --j)
            (*ders)[j].second[0] = ndu(j, deg);

        // Compute the derivatives
        matrix a(deg + 1, deg + 1);
        for (r = 0; r <= deg; r++) {
            int s1, s2;
            s1 = 0;
            s2 = 1; // alternate rows in array a
            a(0, 0) = 1.0;
            // Compute the kth derivative
            for (int k = 1; k <= i; k++) {
                T d;
                int rk, pk, j1, j2;
                d = 0.0;
                rk = r - k;
                pk = deg - k;

                if (r >= k) {
                    a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                    d = a(s2, 0) * ndu(rk, pk);
                }

                if (rk >= -1) {
                    j1 = 1;
                } else {
                    j1 = -rk;
                }

                if (r - 1 <= pk) {
                    j2 = k - 1;
                } else {
                    j2 = deg - r;
                }

                for (j = j1; j <= j2; j++) {
                    a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                    d += a(s2, j) * ndu(rk + j, pk);
                }

                if (r <= pk) {
                    a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                    d += a(s2, k) * ndu(r, pk);
                }
                (*ders)[r].second[k] = d;
                j = s1;
                s1 = s2;
                s2 = j; // Switch rows
            }
        }

        // Multiply through by the correct factors
        r = deg;
        for (int k = 1; k <= i; k++) {
            for (j = deg; j >= 0; --j)
                (*ders)[j].second[k] *= r;
            r *= deg - k;
        }
        delete[] left;
        return ders;
    }

    BasisFunValPac_ptr Eval(const T &u, const unsigned i = 0) const {
        const unsigned dof = GetDof();
        const unsigned deg = GetDegree();
        matrix ders;
        T *left = new T[2 * (deg + 1)];
        T *right = &left[deg + 1];
        matrix ndu(deg + 1, deg + 1);
        T saved, temp;
        int j, r;
        unsigned span = FindSpan(u);
        ders.resize(i + 1, deg + 1);

        ndu(0, 0) = 1.0;
        for (j = 1; j <= deg; j++) {
            left[j] = u - _basisKnot[span + 1 - j];
            right[j] = _basisKnot[span + j] - u;
            saved = 0.0;

            for (r = 0; r < j; r++) {
                // Lower triangle
                ndu(j, r) = right[r + 1] + left[j - r];
                temp = ndu(r, j - 1) / ndu(j, r);
                // _basisKnotpper triangle
                ndu(r, j) = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            ndu(j, j) = saved;
        }

        for (j = deg; j >= 0; --j)
            ders(0, j) = ndu(j, deg);

        // Compute the derivatives
        matrix a(deg + 1, deg + 1);
        for (r = 0; r <= deg; r++) {
            int s1, s2;
            s1 = 0;
            s2 = 1; // alternate rows in array a
            a(0, 0) = 1.0;
            // Compute the kth derivative
            for (int k = 1; k <= i; k++) {
                T d;
                int rk, pk, j1, j2;
                d = 0.0;
                rk = r - k;
                pk = deg - k;

                if (r >= k) {
                    a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                    d = a(s2, 0) * ndu(rk, pk);
                }

                if (rk >= -1) {
                    j1 = 1;
                } else {
                    j1 = -rk;
                }

                if (r - 1 <= pk) {
                    j2 = k - 1;
                } else {
                    j2 = deg - r;
                }

                for (j = j1; j <= j2; j++) {
                    a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                    d += a(s2, j) * ndu(rk + j, pk);
                }

                if (r <= pk) {
                    a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                    d += a(s2, k) * ndu(r, pk);
                }
                ders(k, r) = d;
                j = s1;
                s1 = s2;
                s2 = j; // Switch rows
            }
        }

        // Multiply through by the correct factors
        r = deg;
        for (int k = 1; k <= i; k++) {
            for (j = deg; j >= 0; --j)
                ders(k, j) *= r;
            r *= deg - k;
        }
        delete[] left;
        BasisFunValPac_ptr result(new BasisFunValPac);
        unsigned firstIndex = FirstActive(u);
        for (unsigned ii = 0; ii != ders.cols(); ++ii) {
            result->push_back(BasisFunVal(firstIndex + ii, ders(i, ii)));
        }
        return result;
    }

    T EvalSingle(const T &u, const unsigned n, const unsigned i = 0);

    vector Support(const unsigned i) const {
        const unsigned deg = GetDegree();
        ASSERT(i < GetDof(), "Invalid index of basis function.");
        vector res(2);
        res << _basisKnot[i], _basisKnot[i + deg + 1];
        return res;
    }

    T DomainStart() const { return _basisKnot[GetDegree()]; }

    T DomainEnd() const { return _basisKnot[GetDof()]; }

    unsigned NumActive() const { return GetDegree() + 1; }

    const KnotVector<T> &Knots() const {
        return _basisKnot;
    }

    bool InDomain(T const &u) const { return ((u >= DomainStart()) && (u <= DomainEnd())); }

    void PrintKnots() const { _basisKnot.printKnotVector(); }

    void PrintUniKnots() const { _basisKnot.printUnique(); }

    unsigned FirstActive(T u) const {
        return (InDomain(u) ? FindSpan(u) - GetDegree() : 0);
    }

    bool IsActive(const unsigned i, const T u) const;

protected:
    KnotVector<T> _basisKnot;


};

template<typename T>
BsplineBasis<T>::BsplineBasis() {

}

template<typename T>
BsplineBasis<T>::~BsplineBasis() {

}

template<typename T>
BsplineBasis<T>::BsplineBasis(KnotVector<T> target):_basisKnot(target) {

}

template<typename T>
unsigned BsplineBasis<T>::GetDegree() const {
    return _basisKnot.GetDegree();
}

template<typename T>
unsigned BsplineBasis<T>::GetDof() const {
    return _basisKnot.GetSize() - _basisKnot.GetDegree() - 1;
}

template<typename T>
unsigned BsplineBasis<T>::FindSpan(const T &u) const {
    return _basisKnot.FindSpan(u);
}

template<typename T>
bool BsplineBasis<T>::IsActive(const unsigned i, const T u) const {
    vector supp = Support(i);
    return (u >= supp(0)) && (u < supp(1)) ? true : false;
}

template<typename T>
T BsplineBasis<T>::EvalSingle(const T &u, const unsigned n, const unsigned int i) {
    unsigned p = GetDegree();
    T *ders;
    T **N;
    T *ND;
    N = new T *[p + 1];
    for (int k = 0; k < p + 1; k++)
        N[k] = new T[p + 1];
    ND = new T[i + 1];
    ders = new T[i + 1];
    if (u < _basisKnot[n] || u >= _basisKnot[n + p + 1]) {
        for (int k = 0; k <= i; k++)
            ders[k] = 0;
        double der = ders[i];
        delete[] ders;
        for (int k = 0; k < p + 1; k++)
            delete N[k];
        delete[] N;
        delete[] ND;
        return der;
    }
    for (int j = 0; j <= p; j++) {
        if (u >= _basisKnot[n + j] && u < _basisKnot[n + j + 1])
            N[j][0] = 1;
        else
            N[j][0] = 0;
    }
    double saved;
    for (int k = 1; k <= p; k++) {
        if (N[0][k - 1] == 0.0)
            saved = 0;
        else
            saved = ((u - _basisKnot[n]) * N[0][k - 1]) / (_basisKnot[n + k] - _basisKnot[n]);
        for (int j = 0; j < p - k + 1; j++) {
            double _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + k + 1];
            if (N[j + 1][k - 1] == 0) {
                N[j][k] = saved;
                saved = 0;
            } else {
                double temp = 0;
                if (_basisKnotright != _basisKnotleft)
                    temp = N[j + 1][k - 1] / (_basisKnotright - _basisKnotleft);
                N[j][k] = saved + (_basisKnotright - u) * temp;
                saved = (u - _basisKnotleft) * temp;
            }
        }
    }
    ders[0] = N[0][p];
    for (int k = 1; k <= i; k++) {
        for (int j = 0; j <= k; j++)
            ND[j] = N[j][p - k];
        for (int jj = 1; jj <= k; jj++) {
            if (ND[0] == 0.0)
                saved = 0;
            else
                saved = ND[0] / (_basisKnot[n + p - k + jj] - _basisKnot[n]);
            for (int j = 0; j < k - jj + 1; j++) {
                double _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + p + 1];
                if (ND[j + 1] == 0) {
                    ND[j] = (p - k + jj) * saved;
                    saved = 0;
                } else {
                    double temp = 0;
                    if (_basisKnotright != _basisKnotleft)
                        temp = ND[j + 1] / (_basisKnotright - _basisKnotleft);
                    ND[j] = (p - k + jj) * (saved - temp);
                    saved = temp;
                }
            }
        }
        ders[k] = ND[0];
    }
    double der = ders[i];
    delete[] ders;
    for (int k = 0; k < p + 1; k++)
        delete N[k];
    delete[] N;
    delete[] ND;
    return der;
}


#endif //OO_IGA_BSPLINEBASIS_H
