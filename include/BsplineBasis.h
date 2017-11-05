//
// Created by miaodi on 25/12/2016.
//

#pragma once

#include "KnotVector.h"
#include <memory>
#include "Utility.hpp"
#include "QuadratureRule.h"

template <typename T = double>
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

    virtual ~BsplineBasis();

    int FindSpan(const T &u) const;

    BasisFunValDerAllList_ptr EvalDerAll(const T &u, int i) const
    {
        const int deg = GetDegree();
        BasisFunValDerAll aaa{0, std::vector<T>(i + 1, 0)};
        BasisFunValDerAllList_ptr ders(new BasisFunValDerAllList(deg + 1, aaa));
        T *left = new T[2 * (deg + 1)];
        T *right = &left[deg + 1];
        matrix ndu(deg + 1, deg + 1);
        T saved, temp;
        int j, r;
        int span = FindSpan(u);
        ndu(0, 0) = 1.0;
        for (j = 1; j <= deg; j++)
        {
            left[j] = u - _basisKnot[span + 1 - j];
            right[j] = _basisKnot[span + j] - u;
            saved = 0.0;

            for (r = 0; r < j; r++)
            {
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
        for (r = 0; r <= deg; r++)
        {
            int s1, s2;
            s1 = 0;
            s2 = 1; // alternate rows in array a
            a(0, 0) = 1.0;
            // Compute the kth derivative
            for (int k = 1; k <= i; k++)
            {
                T d;
                int rk, pk, j1, j2;
                d = 0.0;
                rk = r - k;
                pk = deg - k;

                if (r >= k)
                {
                    a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                    d = a(s2, 0) * ndu(rk, pk);
                }

                if (rk >= -1)
                {
                    j1 = 1;
                }
                else
                {
                    j1 = -rk;
                }

                if (r - 1 <= pk)
                {
                    j2 = k - 1;
                }
                else
                {
                    j2 = deg - r;
                }

                for (j = j1; j <= j2; j++)
                {
                    a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                    d += a(s2, j) * ndu(rk + j, pk);
                }

                if (r <= pk)
                {
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
        for (int k = 1; k <= i; k++)
        {
            for (j = deg; j >= 0; --j)
                (*ders)[j].second[k] *= r;
            r *= deg - k;
        }
        delete[] left;

        int firstIndex = FirstActive(u);
        for (int ii = 0; ii != ders->size(); ++ii)
        {
            (*ders)[ii].first = firstIndex + ii;
        }
        return ders;
    }

    BasisFunValPac_ptr Eval(const T &u, const int i = 0) const
    {
        const int dof = GetDof();
        const int deg = GetDegree();
        matrix ders;
        T *left = new T[2 * (deg + 1)];
        T *right = &left[deg + 1];
        matrix ndu(deg + 1, deg + 1);
        T saved, temp;
        int j, r;
        int span = FindSpan(u);
        ders.resize(i + 1, deg + 1);

        ndu(0, 0) = 1.0;
        for (j = 1; j <= deg; j++)
        {
            left[j] = u - _basisKnot[span + 1 - j];
            right[j] = _basisKnot[span + j] - u;
            saved = 0.0;

            for (r = 0; r < j; r++)
            {
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
        for (r = 0; r <= deg; r++)
        {
            int s1, s2;
            s1 = 0;
            s2 = 1; // alternate rows in array a
            a(0, 0) = 1.0;
            // Compute the kth derivative
            for (int k = 1; k <= i; k++)
            {
                T d;
                int rk, pk, j1, j2;
                d = 0.0;
                rk = r - k;
                pk = deg - k;

                if (r >= k)
                {
                    a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                    d = a(s2, 0) * ndu(rk, pk);
                }

                if (rk >= -1)
                {
                    j1 = 1;
                }
                else
                {
                    j1 = -rk;
                }

                if (r - 1 <= pk)
                {
                    j2 = k - 1;
                }
                else
                {
                    j2 = deg - r;
                }

                for (j = j1; j <= j2; j++)
                {
                    a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                    d += a(s2, j) * ndu(rk + j, pk);
                }

                if (r <= pk)
                {
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
        for (int k = 1; k <= i; k++)
        {
            for (j = deg; j >= 0; --j)
                ders(k, j) *= r;
            r *= deg - k;
        }
        delete[] left;
        BasisFunValPac_ptr result(new BasisFunValPac);
        int firstIndex = FirstActive(u);
        for (int ii = 0; ii != ders.cols(); ++ii)
        {
            result->push_back(BasisFunVal(firstIndex + ii, ders(i, ii)));
        }
        return result;
    }

    T EvalSingle(const T &u, const int n, const int i = 0);

    vector Support(const int i) const
    {
        const int deg = GetDegree();
        ASSERT(i < GetDof(), "Invalid index of basis function.");
        vector res(2);
        res << _basisKnot[i], _basisKnot[i + deg + 1];
        return res;
    }

    vector InSpan(const T &u) const
    {
        auto span = FindSpan(u);
        vector res(2);
        res << _basisKnot[span], _basisKnot[span + 1];
        return res;
    }

    T DomainStart() const { return _basisKnot[GetDegree()]; }

    T DomainEnd() const { return _basisKnot[GetDof()]; }

    int NumActive() const { return GetDegree() + 1; }

    const KnotVector<T> &Knots() const
    {
        return _basisKnot;
    }

    bool InDomain(T const &u) const { return ((u >= DomainStart()) && (u <= DomainEnd())); }

    void PrintKnots() const { _basisKnot.printKnotVector(); }

    void PrintUniKnots() const { _basisKnot.printUnique(); }

    int FirstActive(T u) const
    {
        return (InDomain(u) ? FindSpan(u) - GetDegree() : 0);
    }

    std::unique_ptr<matrix> BasisWeight() const
    {
        using QuadList = typename QuadratureRule<T>::QuadList;
        std::unique_ptr<matrix> result(new matrix);
        int dof = _basisKnot.GetDOF();
        auto spans = _basisKnot.KnotEigenSpans();
        int elements = spans.size();
        int quadratureNum = _basisKnot.GetDegree();
        result->resize(elements, dof);
        result->setZero();
        QuadratureRule<T> quadrature(quadratureNum);
        int num = 0;
        for (auto &i : spans)
        {
            QuadList quadList;
            quadrature.MapToQuadrature(i, quadList);
            for (auto &j : quadList)
            {
                auto evals = EvalDerAll(j.first(0), 0);
                for (auto &k : *evals)
                {
                    (*result)(num, k.first) += j.second * k.second[0];
                }
            }
            num++;
        }
        vector sumWeight(dof);
        sumWeight.setZero();
        for (int i = 0; i < dof; i++)
        {
            sumWeight(i) = result->col(i).sum();
        }
        for (int i = 0; i < dof; i++)
        {
            result->col(i) /= sumWeight(i);
        }
        return result;
    }

    bool IsActive(const int i, const T u) const;

    void BezierDualInitialize()
    {
        int degree = _basisKnot.GetDegree();
        _gramianInv = Accessory::GramianInverse<T>(degree);
        _basisWeight = *BasisWeight();
        _reconstruction = *Accessory::BezierReconstruction<T>(_basisKnot);
    }

    BasisFunValDerAllList_ptr BezierDual(const T &u) const
    {
        int degree = _basisKnot.GetDegree();
        vector span = InSpan(u);
        T uPara = (u - span(0)) / (span(1) - span(0));
        auto bernstein = Accessory::AllBernstein(degree, uPara);
        Eigen::Map<vector> bernsteinVector(bernstein.data(), bernstein.size());
        int spanNum = _basisKnot.SpanNum(u);
        int firstIndex = FirstActive(u);
        vector weight = _basisWeight.block(spanNum, firstIndex, 1, degree + 1).transpose();
        vector dual = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(weight.asDiagonal()) * _reconstruction[spanNum].transpose() * _gramianInv *
                      bernsteinVector / (span(1) - span(0));
        BasisFunValDerAll aaa{0, std::vector<T>(1, 0)};
        BasisFunValDerAllList_ptr result(new BasisFunValDerAllList(degree + 1, aaa));
        for (int ii = 0; ii != result->size(); ii++)
            (*result)[ii].second[0] = dual(ii);
        for (int ii = 0; ii != result->size(); ++ii)
        {
            (*result)[ii].first = firstIndex + ii;
        }
        return result;
    }

  protected:
    KnotVector<T> _basisKnot;
    std::vector<matrix> _reconstruction;
    matrix _basisWeight;
    matrix _gramianInv;
};

template <typename T>
BsplineBasis<T>::BsplineBasis()
{
}

template <typename T>
BsplineBasis<T>::~BsplineBasis()
{
}

template <typename T>
BsplineBasis<T>::BsplineBasis(KnotVector<T> target) : _basisKnot(target)
{
}

template <typename T>
int BsplineBasis<T>::GetDegree() const
{
    return _basisKnot.GetDegree();
}

template <typename T>
int BsplineBasis<T>::GetDof() const
{
    return _basisKnot.GetSize() - _basisKnot.GetDegree() - 1;
}

template <typename T>
int BsplineBasis<T>::FindSpan(const T &u) const
{
    return _basisKnot.FindSpan(u);
}

template <typename T>
bool BsplineBasis<T>::IsActive(const int i, const T u) const
{
    vector supp = Support(i);
    return (u >= supp(0)) && (u < supp(1)) ? true : false;
}

template <typename T>
T BsplineBasis<T>::EvalSingle(const T &u, const int n, const int i)
{
    int p = GetDegree();
    T *ders;
    T **N;
    T *ND;
    N = new T *[p + 1];
    for (int k = 0; k < p + 1; k++)
        N[k] = new T[p + 1];
    ND = new T[i + 1];
    ders = new T[i + 1];
    if (u < _basisKnot[n] || u >= _basisKnot[n + p + 1])
    {
        for (int k = 0; k <= i; k++)
            ders[k] = 0;
        T der = ders[i];
        delete[] ders;
        for (int k = 0; k < p + 1; k++)
            delete N[k];
        delete[] N;
        delete[] ND;
        return der;
    }
    for (int j = 0; j <= p; j++)
    {
        if (u >= _basisKnot[n + j] && u < _basisKnot[n + j + 1])
            N[j][0] = 1;
        else
            N[j][0] = 0;
    }
    T saved;
    for (int k = 1; k <= p; k++)
    {
        if (N[0][k - 1] == 0.0)
            saved = 0;
        else
            saved = ((u - _basisKnot[n]) * N[0][k - 1]) / (_basisKnot[n + k] - _basisKnot[n]);
        for (int j = 0; j < p - k + 1; j++)
        {
            T _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + k + 1];
            if (N[j + 1][k - 1] == 0)
            {
                N[j][k] = saved;
                saved = 0;
            }
            else
            {
                T temp = 0;
                if (_basisKnotright != _basisKnotleft)
                    temp = N[j + 1][k - 1] / (_basisKnotright - _basisKnotleft);
                N[j][k] = saved + (_basisKnotright - u) * temp;
                saved = (u - _basisKnotleft) * temp;
            }
        }
    }
    ders[0] = N[0][p];
    for (int k = 1; k <= i; k++)
    {
        for (int j = 0; j <= k; j++)
            ND[j] = N[j][p - k];
        for (int jj = 1; jj <= k; jj++)
        {
            if (ND[0] == 0.0)
                saved = 0;
            else
                saved = ND[0] / (_basisKnot[n + p - k + jj] - _basisKnot[n]);
            for (int j = 0; j < k - jj + 1; j++)
            {
                T _basisKnotleft = _basisKnot[n + j + 1], _basisKnotright = _basisKnot[n + j + p + 1];
                if (ND[j + 1] == 0)
                {
                    ND[j] = (p - k + jj) * saved;
                    saved = 0;
                }
                else
                {
                    T temp = 0;
                    if (_basisKnotright != _basisKnotleft)
                        temp = ND[j + 1] / (_basisKnotright - _basisKnotleft);
                    ND[j] = (p - k + jj) * (saved - temp);
                    saved = temp;
                }
            }
        }
        ders[k] = ND[0];
    }
    T der = ders[i];
    delete[] ders;
    for (int k = 0; k < p + 1; k++)
        delete N[k];
    delete[] N;
    delete[] ND;
    return der;
}
