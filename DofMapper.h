//
// Created by di miao on 2017/6/9.
//

#ifndef OO_IGA_DOFMAPPER_H
#define OO_IGA_DOFMAPPER_H

#include "Visitor.h"
#include <set>
template<typename T>
class Element;

template<typename T>
class Visitor;

template<typename T>
class DofMapper {
public:
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;
    using IndexedValue = typename Visitor<T>::IndexedValue;
    using IndexedValueList = typename Visitor<T>::IndexedValueList;

public:
    DofMapper() {};

    void DomainLabel(DomainShared_ptr domain) {
        if (std::find(_domains.begin(), _domains.end(), domain) == _domains.end()) {
            _domains.push_back(domain);
        }
    }

    int DomainIndex(DomainShared_ptr domain) const {
        int res = std::find(_domains.begin(), _domains.end(), domain) - _domains.begin();
        ASSERT(res < _domains.size(), "unknown domain. You should insert it first.");
        return res;
    }

    void PatchDofSetter(DomainShared_ptr domain, int num) {
        ASSERT(_patchDof.find(domain) == _patchDof.end(),
               "Already initialized");
        _patchDof[domain] = num;
    }

    void FreezedDofInserter(DomainShared_ptr domain, int num) {
        _freezeDof[domain].insert(num);
    }

    int StartingIndex(DomainShared_ptr domain) const {
        int index = DomainIndex(domain);
        int res = 0;
        for (int i = 0; i != index; ++i) {
            res += _patchDof.at(_domains[i]);
        }
        return res;
    }

    int FreeStartingIndex(DomainShared_ptr domain) const {
        int res = StartingIndex(domain);
        int index = DomainIndex(domain);
        for (int i = 0; i != index; ++i) {
            res -= _freezeDof.at(_domains[i]).size();
        }
        return res;
    }

    int Dof() const{
        int res = 0;
        for (const auto &i:_domains) {
            res += _patchDof.at(i);
        }
        return res;
    }
    int FreeDof() const{
        int res = 0;
        for (const auto &i:_domains) {
            res += _patchDof.at(i);
            res -= _freezeDof.at(i).size();
        }
        return res;
    }
private:
    std::vector<DomainShared_ptr> _domains;
    std::map<DomainShared_ptr, int> _patchDof;
    std::map<DomainShared_ptr, std::set<int>> _freezeDof;

};


#endif //OO_IGA_DOFMAPPER_H
