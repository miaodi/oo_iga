//
// Created by miaodi on 23/10/2017.
//

#pragma once

#include <boost/math/special_functions/binomial.hpp>
#include <eigen3/Eigen/Dense>
#include <set>
#include <vector>
#include <map>
#include <iostream>
#include <eigen3/Eigen/Sparse>

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)
#else
#define ASSERT(condition, message) \
    do                             \
    {                              \
    } while (false)
#endif

template <int d, int N, typename T>
class PhyTensorBsplineBasis;

template <typename T>
class KnotVector;

namespace Accessory
{
using namespace Eigen;
template <typename T, int N>
using ContPtsList = std::vector<Eigen::Matrix<T, N, 1>>;

using DifferentialPattern = std::vector<int>;
using DifferentialPatternList = std::vector<DifferentialPattern>;
using DifferentialPatternList_ptr = std::unique_ptr<DifferentialPatternList>;
template <typename T>
using ExtractionOperator = Matrix<T, Dynamic, Dynamic>;
template <typename T>
using ExtractionOperatorContainer = std::vector<ExtractionOperator<T>>;

template <typename T>
void binomialCoef(Matrix<T, Dynamic, Dynamic> &Bin)
{
    int n, k;
    // Setup the first line
    Bin(0, 0) = T(1);
    for (k = static_cast<int>(Bin.cols()) - 1; k > 0; --k)
        Bin(0, k) = T(0);
    // Setup the other lines
    for (n = 0; n < static_cast<int>(Bin.rows()) - 1; n++)
    {
        Bin(n + 1, 0) = T(1);
        for (k = 1; k < static_cast<int>(Bin.cols()); k++)
            if (n + 1 < k)
                Bin(n, k) = T(0);
            else
                Bin(n + 1, k) = Bin(n, k) + Bin(n, k - 1);
    }
}

template <typename T, int N>
void degreeElevate(int t,
                   KnotVector<T> &U,
                   ContPtsList<T, N> &P)
{
    ASSERT(t > 0,
           "Invalid geometrical information input, check size bro.");
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
    std::vector<Matrix<T, N, 1>> bpts(p + 1);     // pth-degree Bezier control points of the current segment
    std::vector<Matrix<T, N, 1>> ebpts(p + t + 1);         // (p+t)th-degree Bezier control points of the  current segment
    std::vector<Matrix<T, N, 1>> Nextbpts(p - 1);          // leftmost control points of the next Bezier segment
    std::vector<T> alphas(p - 1, T(0));                       // knot instertion alphas.
    // Compute the binomial coefficients
    Matrix<T, Dynamic, Dynamic> Bin(ph + 1, ph2 + 1);
    bezalfs.setZero();
    Bin.setZero();
    binomialCoef(Bin);
    // Compute Bezier degree elevation coefficients
    T inv; 
    int mpi;
    bezalfs(0, 0) = bezalfs(ph, p) = T(1);
    for (i = 1; i <= ph2; i++)
    {
        inv = T(1) / Bin(ph, i);
        mpi = std::min(p, i);
        for (j = std::max(0, i - t);
             j <= mpi; j++)
        {
            bezalfs(i, j) = inv * Bin(p, j) * Bin(t, i - j);
        }
    }
    
    for (i = ph2 + 1; i < ph; i++)
    {
        mpi = std::min(p, i);
        for (j = std::max(0, i - t); j <= mpi; j++)
            bezalfs(i, j) = bezalfs(ph - i, p - j);
    }

    P.resize(cP.size() * t * 3); // Allocate more control points than necessary
    U.resize(cP.size() * t * 3 + ph + 1);
    int mh = ph;
    int kind = ph + 1;
    T ua = U(0);
    T ub = U(0);
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
    for (i = 0; i <= ph; i++)
    {
        U(i) = ua;
    }

    // Initialize the first Bezier segment
    for (i = 0; i <= p; i++)
        bpts[i] = cP[i];
    while (b < m)
    { // Big loop thru knot vector
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
        if (r > 0)
        { // Insert knot to get Bezier segment
            numer = ub - ua;
            for (k = p; k > mul; k--)
            {
                alphas[k - mul - 1] = numer / (cU(a + k) - ua);
            }
            for (j = 1; j <= r; j++)
            {
                save = r - j;
                s = mul + j;
                for (k = p; k >= s; k--)
                {
                    bpts[k] = alphas[k - s] * bpts[k] + (T(1) - alphas[k - s]) * bpts[k - 1];
                }
                Nextbpts[save] = bpts[p];
            }
        }

        for (i = lbz; i <= ph; i++)
        { // Degree elevate Bezier,  only the points lbz,...,ph are used
            ebpts[i] = Matrix<T, N, 1>::Zero(N);
            mpi = std::min(p, i);
            for (j = std::max(0, i - t); j <= mpi; j++)
                ebpts[i] += bezalfs(i, j) * bpts[j];
        }

        if (oldr > 1)
        { // Must remove knot u=c.U[a] oldr times
            // if(oldr>2) // Alphas on the right do not change
            //	alfj = (ua-U[kind-1])/(ub-U[kind-1]) ;
            first = kind - 2;
            last = kind;
            den = ub - ua;
            bet = (ub - U(kind - 1)) / den;
            for (int tr = 1; tr < oldr; tr++)
            { // Knot removal loop
                i = first;
                j = last;
                kj = j - kind + 1;
                while (j - i > tr)
                { // Loop and compute the new control points for one removal step
                    if (i < cind)
                    {
                        alf = (ub - U(i)) / (ua - U(i));
                        P[i] = alf * P[i] + (T(1) - alf) * P[i - 1];
                    }
                    if (j >= lbz)
                    {
                        if (j - tr <= kind - ph + oldr)
                        {
                            gam = (ub - U(j - tr)) / den;
                            ebpts[kj] = gam * ebpts[kj] + (T(1) - gam) * ebpts[kj + 1];
                        }
                        else
                        {
                            ebpts[kj] = bet * ebpts[kj] + (T(1) - bet) * ebpts[kj + 1];
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
            for (i = 0; i < ph - oldr; i++)
            {
                U(kind++) = ua;
            }
        for (j = lbz; j <= rbz; j++)
        { // load control points onto the curve
            P[cind++] = ebpts[j];
        }

        if (b < m)
        { // Set up for next pass thru loop
            for (j = 0; j < r; j++)
                bpts[j] = Nextbpts[j];
            for (j = r; j <= p; j++)
                bpts[j] = cP[b - p + j];
            a = b;
            b++;
            ua = ub;
        }
        else
        {
            for (i = 0; i <= ph; i++)
                U(kind + i) = ub;
        }
    }
    P.resize(mh - ph); // Resize to the proper number of control points
    U.resize(mh + 1);
}

template <typename T, int N>
void knotInsertion(T u,
                   int r,
                   KnotVector<T> &U,
                   ContPtsList<T, N> &P)
{

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
    for (j = 1; j <= r; j++)
    {
        L = k - p + j;
        for (i = 0; i <= p - j - s; i++)
        {
            alpha = (u - cU(L + i)) / (cU(i + k + 1) - cU(L + i));
            R[i] = alpha * R[i + 1] + (T(1) - alpha) * R[i];
        }
        P[L] = R[0];
        P[k + r - j - s] = R[p - j - s];
    }
    for (i = L + 1; i < k - s; i++)
        P[i] = R[i - L];
}

template <typename T, int N>
void refineKnotVectorCurve(const KnotVector<T> &X,
                           KnotVector<T> &U,
                           ContPtsList<T, N> &P)
{

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
    for (j = r; j >= 0; j--)
    {
        while (X[j] <= cU[i] && i > a)
        {
            P[k - p - 1] = cP[i - p - 1];
            U(k) = cU(i);
            --k;
            --i;
        }
        P[k - p - 1] = P[k - p];
        for (int l = 1; l <= p; l++)
        {
            int ind = k - p + l;
            T alpha = U[k + l] - X[j];
            if (alpha == T(0))
                P[ind - 1] = P[ind];
            else
                alpha /= U(k + l) - cU(i - p + l);
            P[ind - 1] = alpha * P[ind - 1] + (T(1) - alpha) * P[ind];
        }
        U(k) = X[j];
        --k;
    }
}

template <int N>
DifferentialPatternList_ptr
PartialDerPattern(int r)
{
    std::vector<int> kk(r);
    DifferentialPatternList_ptr a(new DifferentialPatternList);
    std::function<void(int,
                       int,
                       std::vector<int> &,
                       int,
                       int,
                       DifferentialPatternList_ptr &)>
        recursive;
    recursive = [&](int D,
                    int i,
                    std::vector<int> &k,
                    int n,
                    int start,
                    std::unique_ptr<std::vector<std::vector<int>>> &a) {
        if (n == i)
        {
            std::vector<int> m;
            int it = 0;
            for (int it1 = 0; it1 < D; ++it1)
            {
                int amount = 0;
                while (find(k.begin(),
                            k.end(),
                            it) != k.end())
                {
                    amount++;
                    it++;
                }
                m.push_back(amount);
                it++;
            }
            a->push_back(m);
        }
        else
        {
            for (int jj = start; jj < D + i - (i - n); ++jj)
            {
                k[n] = jj;
                recursive(D,
                          i,
                          k,
                          n + 1,
                          jj + 1,
                          a);
            }
        }
    };
    recursive(N,
              r,
              kk,
              0,
              0,
              a);
    return a;
}

template <typename T>
std::unique_ptr<ExtractionOperatorContainer<T>> BezierExtraction(const KnotVector<T> &knot)
{
    std::unique_ptr<ExtractionOperatorContainer<T>> result(new ExtractionOperatorContainer<T>);
    int m = knot.GetSize();
    int p = knot.GetDegree();
    int a = p + 1;
    int b = a + 1;
    int nb = 1;
    std::vector<T> alphas(p + 1);
    ExtractionOperator<T> C = ExtractionOperator<T>::Identity(p + 1,
                                                              p + 1);
    while (b < m)
    {
        ExtractionOperator<T> C_next = ExtractionOperator<T>::Identity(p + 1,
                                                                       p + 1);
        int i = b;
        while (b < m && knot[b] == knot[b - 1])
        {
            b++;
        }
        int mult = b - i + 1;
        if (mult < p)
        {
            T numer = knot[b - 1] - knot[a - 1];
            for (int j = p; j > mult; j--)
            {
                alphas[j - mult - 1] = numer / (knot[a + j - 1] - knot[a - 1]);
            }
            int r = p - mult;
            for (int j = 1; j <= r; j++)
            {
                int save = r - j + 1;
                int s = mult + j;
                for (int k = p + 1; k >= s + 1; k--)
                {
                    T alpha = alphas[k - s - 1];
                    C.col(k - 1) = alpha * C.col(k - 1) + (T(1) - alpha) * C.col(k - 2);
                }

                if (b < m)
                {
                    for (int l = 0; l <= j; l++)
                    {
                        C_next(save + l - 1,
                               save - 1) = C(p - j + l,
                                             p);
                    }
                }
            }
            result->push_back(C);
            C = C_next;
            nb++;
            if (b < m)
            {
                a = b;
                b++;
            }
        }
    }
    result->push_back(C);
    return result;
}

template <typename T>
std::unique_ptr<ExtractionOperatorContainer<T>> BezierReconstruction(const KnotVector<T> &knot)
{
    auto res = BezierExtraction<T>(knot);
    for (auto &i : *res)
    {
        i = i.inverse();
    }
    return res;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> Gramian(int p)
{
    using namespace boost::math;
    int n = p + 1;
    Matrix<T, Dynamic, Dynamic> res(n,
                                    n);
    res.setZero();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            res(i,
                j) = binomial_coefficient<T>(p,
                                             i) *
                     binomial_coefficient<T>(p,
                                             j) /
                     (2 * p + 1) /
                     binomial_coefficient<T>(2 * p,
                                             i + j);
        }
    }
    res = res.template selfadjointView<Eigen::Lower>();
    return res;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> GramianInverse(int p)
{
    using namespace boost::math;
    int n = p + 1;
    Matrix<T, Dynamic, Dynamic> res(n,
                                    n);
    res.setZero();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            T sum = 0;
            for (int k = 0;
                 k <= std::min(i,
                               j);
                 k++)
            {
                sum += (2 * k + 1) * binomial_coefficient<T>(p + k + 1,
                                                             p - i) *
                       binomial_coefficient<T>(p - k,
                                               p - i) *
                       binomial_coefficient<T>(p + k + 1,
                                               p - j) *
                       binomial_coefficient<T>(p - k,
                                               p - j);
            }
            res(i,
                j) = sum * pow(-1,
                               i + j) /
                     binomial_coefficient<T>(p,
                                             i) /
                     binomial_coefficient<T>(p,
                                             j);
        }
    }
    res = res.template selfadjointView<Eigen::Lower>();
    return res;
}

template <typename T>
std::vector<T> AllBernstein(int p,
                            T u)
{
    std::vector<T> res(p + 1);
    res[0] = 1;
    T u1 = 1 - u;
    for (int j = 1; j <= p; j++)
    {
        T saved = 0;
        for (int k = 0; k < j; k++)
        {
            T temp = res[k];
            res[k] = saved + u1 * temp;
            saved = u * temp;
        }
        res[j] = saved;
    }
    return res;
}

template <int N, int d_from, int d_to, typename T>
bool MapParametricPoint(const PhyTensorBsplineBasis<d_from, N, T> *const from_domain,
                        const Eigen::Matrix<T, Eigen::Dynamic, 1> &from_point,
                        const PhyTensorBsplineBasis<d_to, N, T> *const to_domain,
                        Eigen::Matrix<T, Eigen::Dynamic, 1> &to_point)
{
    ASSERT(from_domain->InDomain(from_point),
           "The point about to be mapped is out of the domain.");
    Eigen::Matrix<T, N, 1> physical_point = from_domain->AffineMap(from_point);
    return to_domain->InversePts(physical_point,
                                 to_point);
}

std::map<int, int> IndicesInverseMap(const std::vector<int> &forward_map);

template <typename T>
std::set<int> ColIndicesSet(const std::vector<Eigen::Triplet<T>> &triplet)
{
    std::set<int> res;
    for (const auto &i : triplet)
    {
        res.insert(i.col());
    }
    return res;
}

template <typename T>
std::vector<int> ColIndicesVector(const std::vector<Eigen::Triplet<T>> &triplet)
{
    std::set<int> res;
    for (const auto &i : triplet)
    {
        res.insert(i.col());
    }
    std::vector<int> res_vector(res.begin(), res.end());
    return res_vector;
}

template <typename T>
std::set<int> RowIndicesSet(const std::vector<Eigen::Triplet<T>> &triplet)
{
    std::set<int> res;
    for (const auto &i : triplet)
    {
        res.insert(i.row());
    }
    return res;
}

template <typename T>
std::vector<int> RowIndicesVector(const std::vector<Eigen::Triplet<T>> &triplet)
{
    std::set<int> res;
    for (const auto &i : triplet)
    {
        res.insert(i.row());
    }
    std::vector<int> res_vector(res.begin(), res.end());
    return res_vector;
}
}
