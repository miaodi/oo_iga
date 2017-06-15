//
// Created by di miao on 2017/6/9.
//

#ifndef OO_IGA_DOFMAPPER_H
#define OO_IGA_DOFMAPPER_H

#include "Visitor.h"
#include <set>
#include <boost/bimap.hpp>

template<typename T>
class Element;

template<typename T>
class Visitor;
namespace Accessory {
    using IndexBiMap = boost::bimap<int, int>;

    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>> BiMapToSparseMatrix(int row, int col,const IndexBiMap &bimap) {
        using IndexedValue = Eigen::Triplet<T>;
        using IndexedValueList = std::vector<IndexedValue>;
        IndexedValueList temp;
        for (auto it = bimap.right.begin(); it != bimap.right.end(); ++it) {
            temp.push_back(IndexedValue(it->second, it->first, 1));
        }
        std::unique_ptr<Eigen::SparseMatrix<T>> res(new Eigen::SparseMatrix<T>);
        res->resize(row, col);
        res->setFromTriplets(temp.begin(),temp.end());
        return res;
    }

}
template<typename T>
class DofMapper {
public:
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;
    using IndexBiMap = boost::bimap<int, int>;
    using IndexPair = IndexBiMap::value_type;

public:
    DofMapper() {};

    //insert a domain pointer into the container.
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

    int Dof() const {
        int res = 0;
        for (const auto &i:_domains) {
            res += _patchDof.at(i);
        }
        return res;
    }

    int FreeDof() const {
        int res = 0;
        for (const auto &i:_domains) {
            res += _patchDof.at(i);
            res -= _freezeDof.at(i).size();
        }
        return res;
    }

    int Index(DomainShared_ptr domain, int i) const {
        return StartingIndex(domain) + i;
    }

    bool FreeIndexInDomain(DomainShared_ptr domain, int &i) const {
        auto domainIndex = DomainIndex(domain);
        auto res = _freezeDof.at(_domains[domainIndex]).find(i) == _freezeDof.at(_domains[domainIndex]).end();
        if (res) {
            i -= std::count_if(_freezeDof.at(_domains[domainIndex]).begin(),
                               _freezeDof.at(_domains[domainIndex]).end(),
                               [&i](int num) { return num < i; });
        }
        return res;
    }

    int FreeIndex(DomainShared_ptr domain, int &i) const {
        auto res = FreeIndexInDomain(domain, i);
        if (res) {
            i += FreeStartingIndex(domain);
        }
        return res;
    }

    IndexBiMap CondensedBiMap() {
        IndexBiMap res;
        for (const auto &i:_domains) {
            auto startIndex = StartingIndex(i);
            auto startFreeIndex = FreeStartingIndex(i);
            for (int inDomainIndex = 0, domainDof = _patchDof[i]; inDomainIndex != domainDof; ++inDomainIndex) {
                int copyIndex = inDomainIndex;
                if (FreeIndexInDomain(i, copyIndex)) {
                    res.insert(IndexPair(startFreeIndex + copyIndex, inDomainIndex + startIndex));
                }
            }
        }
        return res;
    }

    void PrintFreezedDofIn(const DomainShared_ptr domain) {
        auto it = _freezeDof.find(domain);
        std::cout << "Freezed Dof Index in given domain are:";
        for (const auto &i:it->second) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    void PrintDofIn(const DomainShared_ptr domain) {
        auto it = _patchDof.find(domain);
        std::cout << "Dof of given domain are:" << it->second;
        std::cout << std::endl;
    }


    void PrintFreeDofIn(const DomainShared_ptr domain) {
        auto it = _patchDof.find(domain);
        auto itit = _freezeDof.find(domain);
        std::cout << "Free Dof of given domain are:" << it->second-itit->second.size();
        std::cout << std::endl;
    }
private:
    std::vector<DomainShared_ptr> _domains;
    std::map<DomainShared_ptr, int> _patchDof;
    std::map<DomainShared_ptr, std::set<int>> _freezeDof;

};


#endif //OO_IGA_DOFMAPPER_H
