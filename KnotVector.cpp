//
// Created by di miao on 12/23/16.
//

#include "KnotVector.h"

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
    _multiKnots.clear();
    for (auto const &s : _uniKnots) {
        for (unsigned i = 0; i < s.second; ++i) {
            _multiKnots.push_back(s.first);
        }
    }
}

template<typename T>
void KnotVector<T>::UniformRefine(unsigned r, unsigned multi) {
    for (unsigned i = 0; i < r; i++) {
        std::pair<T, unsigned> temp = (*_uniKnots.begin());
        uniContainer tmp;
        for (const auto &e : _uniKnots) {
            tmp.emplace((temp.first + e.first) / 2, multi);
            temp = e;
        }
        _uniKnots.insert(tmp.begin(), tmp.end());
    }
    MultiPle();
}

template<typename T>
void KnotVector<T>::RefineSpan(std::pair<T, T> span, unsigned int r, unsigned int multi) {
    auto itlow = _uniKnots.lower_bound(span.first), itup = _uniKnots.upper_bound(span.second);
    KnotVector<T> temp({itlow, itup});
    temp.UniformRefine(r, multi);
    _uniKnots.insert(temp._uniKnots.begin(), temp._uniKnots.end());
    MultiPle();
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> KnotVector<T>::MapToEigen() const {
    return Eigen::Matrix<T, -1, 1, 0, -1, 1>::Map(_multiKnots.data(), _multiKnots.size());
}

template <typename T>
unsigned KnotVector<T>::GetSize() const {
    return static_cast<unsigned>(_multiKnots.size());
}

template<typename T>
const T& KnotVector<T>::operator[](unsigned i) const {
    return _multiKnots[i];
}


template  class KnotVector<double>;