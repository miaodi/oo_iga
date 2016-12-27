//
// Created by di miao on 12/23/16.
//

#ifndef OO_IGA_KNOTVECTOR_H
#define OO_IGA_KNOTVECTOR_H

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

    KnotVector(const KnotVector &target);

    const T& operator[](unsigned i) const;

    void Insert(T knot);

    void printUnique() const;

    void printKnotVector() const;

    void UniformRefine(unsigned r = 1, unsigned multi = 1);

    void RefineSpan(std::pair<T, T>, unsigned r = 1, unsigned multi = 1);

    Eigen::Matrix<T, Eigen::Dynamic, 1> MapToEigen() const;

    unsigned GetDegree() const;

    unsigned GetSize() const;

private:
    //Knots with repetitions {0,0,0,.5,1,1,1}
    knotContainer _multiKnots;
    //Unique knots associated with multiplicity {0:3,.5:1,1:3}
    uniContainer _uniKnots;
    //


    void UniQue();

    void MultiPle();

};


#endif //OO_IGA_KNOTVECTOR_H
