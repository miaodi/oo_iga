//
// Created by di miao on 12/23/16.
//

#pragma once

#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>
#include <iostream>

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

template <typename T>
class KnotVector
{
  public:
    using knotContainer = std::vector<T>;
    using uniContainer = std::map<T, int>;

    //methods
    KnotVector(){};

    KnotVector(const knotContainer &target);

    KnotVector(const uniContainer &target);

    const T &operator[](int i) const;

    T &operator()(int i);

    void Insert(T knot);

    void printUnique() const;

    knotContainer GetUnique() const;

    void printKnotVector() const;

    void UniformRefine(int r = 1, int multi = 1);

    // Uniformly refine the knot vector such that the d.o.f of the new knot vector is close to the given number
    void RefineToDof(const int &);

    KnotVector RefineToDofKnotVector(const int &) const;

    void RefineSpan(std::pair<T, T>, int r = 1, int multi = 1);

    Eigen::Matrix<T, Eigen::Dynamic, 1> MapToEigen() const;

    int GetDegree() const;

    int GetSize() const;

    int GetSpanSize() const;

    int GetDOF() const;

    void InitClosed(int _deg, T first = T(0.0), T last = T(1.0));

    void InitClosedUniform(int _dof, int _deg, T first = T(0.0), T last = T(1.0));

    KnotVector UniKnotUnion(const KnotVector &vb) const;

    int FindSpan(const T &u) const;

    std::vector<std::pair<T, T>> KnotSpans() const;

    int SpanNum(const T &u) const;

    std::vector<std::pair<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Matrix<T, Eigen::Dynamic, 1>>>
    KnotEigenSpans() const;

    void resize(int t) { _multiKnots.resize(t); };

    KnotVector Difference(const KnotVector &) const;

    // leave the knot vector with unique knots, difference within tolerance will not be considered.
    void Uniquify(const T &tol = 1e-14);

  private:
    //Knots with repetitions {0,0,0,.5,1,1,1}
    knotContainer _multiKnots;

    void UniQue(uniContainer &) const;

    void MultiPle(const uniContainer &);
};