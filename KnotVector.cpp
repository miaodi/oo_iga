//
// Created by di miao on 12/23/16.
//

#include "KnotVector.h"

template<typename T>
KnotVector<T>::KnotVector(const KnotVector::knotContainer &target):_multiKnots(target) {
}

template<typename T>
KnotVector<T>::KnotVector(const KnotVector::uniContainer &target) {
    MultiPle(target);
}


template<typename T>
void KnotVector<T>::UniQue(uniContainer &_uniKnots) const {
    _uniKnots.clear();
    for (auto const &s : _multiKnots) {
        ++_uniKnots[s];
    }
}

template<typename T>
typename KnotVector<T>::knotContainer KnotVector<T>::GetUnique() const {
    knotContainer result;
    uniContainer _uniKnots;
    UniQue(_uniKnots);
    for (auto const &e : _uniKnots) {
        result.push_back(e.first);
    }
    return result;
}


template<typename T>
void KnotVector<T>::printUnique() const {
    uniContainer _uniKnots;
    UniQue(_uniKnots);
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
int KnotVector<T>::GetDegree() const {
    uniContainer _uniKnots;
    UniQue(_uniKnots);
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
}

template<typename T>
void KnotVector<T>::MultiPle(const uniContainer &_uniKnots) {
    _multiKnots.clear();
    for (auto const &s : _uniKnots) {
        for (int i = 0; i < s.second; ++i) {
            _multiKnots.push_back(s.first);
        }
    }
}

template<typename T>
void KnotVector<T>::UniformRefine(int r, int multi) {
    uniContainer _uniKnots;
    UniQue(_uniKnots);
    for (int i = 0; i < r; i++) {
        std::pair<T, int> temp = (*_uniKnots.begin());
        uniContainer tmp;
        for (const auto &e : _uniKnots) {
            tmp.emplace((temp.first + e.first) / 2, multi);
            temp = e;
        }
        _uniKnots.insert(tmp.begin(), tmp.end());
    }
    MultiPle(_uniKnots);
}

template<typename T>
void KnotVector<T>::RefineSpan(std::pair<T, T> span, int r, int multi) {
    uniContainer _uniKnots, temp_uniKnots;
    UniQue(_uniKnots);
    auto itlow = _uniKnots.lower_bound(span.first), itup = _uniKnots.upper_bound(span.second);
    auto spanOnKnot = {*itlow, *itup};
    KnotVector<T> temp(spanOnKnot);
    temp.UniQue(temp_uniKnots);
    temp.UniformRefine(r, multi);
    _uniKnots.insert(temp_uniKnots.begin(), temp_uniKnots.end());
    MultiPle(_uniKnots);
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> KnotVector<T>::MapToEigen() const {
    return Eigen::Matrix<T, -1, 1, 0, -1, 1>::Map(_multiKnots.data(), _multiKnots.size());
}

template<typename T>
int KnotVector<T>::GetSize() const {
    return static_cast<int>(_multiKnots.size());
}

template<typename T>
const T &KnotVector<T>::operator[](int i) const {
    return _multiKnots[i];
}

template<typename T>
void KnotVector<T>::InitClosed(int _deg, T first, T last) {
    uniContainer _uniKnots;
    uniContainer tmp;
    tmp.emplace(first, _deg + 1);
    tmp.emplace(last, _deg + 1);
    _uniKnots.insert(tmp.begin(), tmp.end());
    MultiPle(_uniKnots);
}

template<typename T>
void KnotVector<T>::InitClosedUniform(int _dof, int _deg, T first, T last) {
    InitClosed(_deg, first, last);
    uniContainer _uniKnots;
    UniQue(_uniKnots);
    ASSERT(_dof > _deg + 1, "Degree of freedom is too small.");
    InitClosed(_deg, first, last);
    const T interval = (last - first) / double(_dof - _deg);
    T knot = interval;
    uniContainer tmp;
    for (int i = 1; i < _dof - _deg; ++i) {
        tmp.emplace(knot, 1);
        knot += interval;
    }
    _uniKnots.insert(tmp.begin(), tmp.end());
    MultiPle(_uniKnots);
}

template<typename T>
KnotVector<T> KnotVector<T>::UniKnotUnion(const KnotVector &vb) const {
    uniContainer _uniKnots, vb_uniKnots;
    UniQue(_uniKnots);
    vb.UniQue(vb_uniKnots);
    uniContainer tmp = _uniKnots;
    tmp.insert(vb_uniKnots.begin(), vb_uniKnots.end());
    for (auto &e : tmp) {
        e.second = 1;
    }
    return KnotVector(tmp);
}

template<typename T>
std::vector<std::pair<T, T>> KnotVector<T>::KnotSpans() const {
    uniContainer _uniKnots;
    UniQue(_uniKnots);
    std::vector<std::pair<T, T>> tmp;
    for (auto it = _uniKnots.begin(); it != std::prev(_uniKnots.end()); ++it) {
        tmp.push_back(std::make_pair(it->first, std::next(it, 1)->first));
    }
    return tmp;
}

template<typename T>
int KnotVector<T>::GetDOF() const {
    return GetSize() - GetDegree() - 1;
}


template<typename T>
T &KnotVector<T>::operator()(int i) {
    return _multiKnots[i];
}

template<typename T>
int KnotVector<T>::FindSpan(const T &u) const {
    const int dof = GetDOF();
    const int deg = GetDegree();
    if (u >= _multiKnots[dof])
        return dof - 1;
    if (u <= _multiKnots[deg])
        return deg;

    int low = 0;
    int high = dof + 1;
    int mid = (low + high) / 2;

    while (u < _multiKnots[mid] || u >= _multiKnots[mid + 1]) {
        if (u < _multiKnots[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

template<typename T>
KnotVector<T> KnotVector<T>::Difference(const KnotVector &reference) const {
    knotContainer diff;
    typename knotContainer::iterator it;
    std::set_difference(_multiKnots.begin(), _multiKnots.end(), reference._multiKnots.begin(),
                        reference._multiKnots.end(), std::back_inserter(diff));
    return KnotVector(diff);
}

template<typename T>
std::vector<std::pair<Eigen::Matrix<T, 1, 1>, Eigen::Matrix<T, 1, 1>>>
KnotVector<T>::KnotEigenSpans() const {
    using Coordinate = Eigen::Matrix<T, 1, 1>;
    uniContainer _uniKnots;
    UniQue(_uniKnots);
    std::vector<std::pair<Coordinate, Coordinate>> tmp;
    for (auto it = _uniKnots.begin(); it != std::prev(_uniKnots.end()); ++it) {
        tmp.push_back(std::make_pair(Coordinate(it->first), Coordinate(std::next(it, 1)->first)));
    }
    return tmp;
}


template
class KnotVector<double>;