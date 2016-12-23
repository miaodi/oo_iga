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
    typedef std::vector<T> knotContainer;
    typedef std::map<T, unsigned> uniContainer;

    //methods
    KnotVector() {};

    KnotVector(const knotContainer &target);

    KnotVector(const uniContainer &target);

    KnotVector(const KnotVector &target);

    void Insert(T knot);

    void printUnique() const;

    void printKnotVector() const;

    unsigned GetDegree() const;

private:
    //Knots with repetitions {0,0,0,.5,1,1,1}
    knotContainer _multiKnots;
    //Unique knots associated with multiplicity {0:3,.5:1,1:3}
    uniContainer _uniKnots;
    //


    void UniQue();

    void MultiPle();

};

template<typename T>
KnotVector<T>::KnotVector(const KnotVector::knotContainer &target):_multiKnots(target) {
    UniQue();
}

template<typename T>
KnotVector<T>::KnotVector(const KnotVector::uniContainer &target):_uniKnots(target) {
    MultiPle();
}

template<typename T>
KnotVector<T>::KnotVector(const KnotVector &target):_multiKnots(target._multiKnots), _uniKnots(target._uniKnots) {

}


template<typename T>
void KnotVector<T>::UniQue() {
    _uniKnots.clear();
    for (auto const &s : _multiKnots) {
        ++_uniKnots[s];
    }
}

template<typename T>
void KnotVector<T>::printUnique() const {
    for (auto const &e : _uniKnots) {
        std::cout << e.first << " : " << e.second << std::endl;
    }
}

template<typename T>
void KnotVector<T>::printKnotVector() const {
    for (auto const &e : _multiKnots) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

template<typename T>
unsigned KnotVector<T>::GetDegree() const {
    return (*_uniKnots.begin()).second - 1;
}

template<typename T>
void KnotVector<T>::Insert(T r) {
    for (auto it = _multiKnots.begin() + 1; it != _multiKnots.end(); ++it) {
        if (r <= *it && r >= *(it - 1)) {
            _multiKnots.emplace(it, r);
            break;
        }
    }
    UniQue();
}

template<typename T>
void KnotVector<T>::MultiPle() {
    for (auto const &s : _uniKnots) {
        for (unsigned i = 0; i < s.second; ++i) {
            _multiKnots.push_back(s.first);
        }
    }
}


#endif //OO_IGA_KNOTVECTOR_H
