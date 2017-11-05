//
// Created by miaodi on 25/12/2016.
//

#include "BsplineBasis.h"
#include <boost/multiprecision/gmp.hpp>

template <typename T>
BsplineBasis<T>::BsplineBasis()
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


template class BsplineBasis<double>;
template class BsplineBasis<float>;
template class BsplineBasis<boost::multiprecision::mpf_float_50>;
template class BsplineBasis<boost::multiprecision::mpf_float_100>;
template class BsplineBasis<boost::multiprecision::mpf_float_1000>;
