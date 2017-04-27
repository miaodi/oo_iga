//
// Created by di miao on 12/23/16.
//

#ifndef OO_IGA_KNOTVECTOR_H
#define OO_IGA_KNOTVECTOR_H
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
#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>
#include <iostream>

template<typename T>
class KnotVector {
public:
    using knotContainer = std::vector<T>;
    using uniContainer = std::map<T, unsigned>;

    //methods
    KnotVector() {};

    KnotVector(const knotContainer &target);

    KnotVector(const uniContainer &target);

    const T& operator[](unsigned i) const;

    T& operator()(unsigned i);

    T& operator()(int i);

    void Insert(T knot);

    void printUnique() const;

    void printKnotVector() const;

    void UniformRefine(unsigned r = 1, unsigned multi = 1);

    void RefineSpan(std::pair<T, T>, unsigned r = 1, unsigned multi = 1);

    Eigen::Matrix<T, Eigen::Dynamic, 1> MapToEigen() const;

    unsigned GetDegree() const;

    unsigned GetSize() const;

    unsigned GetDOF() const;

    void InitClosed(unsigned _deg, T first=T(0.0), T last=T(1.0));

    void InitClosedUniform(unsigned _dof, unsigned _deg, T first=T(0.0), T last=T(1.0));

    KnotVector UniKnotUnion(const KnotVector & vb) const;

    unsigned FindSpan(const T &u) const;

    std::vector<std::pair<T,T>> KnotSpans() const;

    void resize(unsigned t){_multiKnots.resize(t);};

    KnotVector Difference(const KnotVector&) const;

private:
    //Knots with repetitions {0,0,0,.5,1,1,1}
    knotContainer _multiKnots;

    void UniQue(uniContainer &) const;

    void MultiPle(const uniContainer &);

};


#endif //OO_IGA_KNOTVECTOR_H
