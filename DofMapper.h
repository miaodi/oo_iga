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
namespace Accessory {
    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>> SparseMatrixMaker(const std::vector<Eigen::Triplet<T>> &_list) {
        using IndexedValue = Eigen::Triplet<T>;
        using IndexedValueList = std::vector<IndexedValue>;
        std::unique_ptr<Eigen::SparseMatrix<T>> matrix(new Eigen::SparseMatrix<T>);
        auto row = std::max_element(_list.begin(), _list.end(),
                                    [](const IndexedValue &a, const IndexedValue &b) -> bool {
                                        return a.row() < b.row();
                                    });
        auto col = std::max_element(_list.begin(), _list.end(),
                                    [](const IndexedValue &a, const IndexedValue &b) -> bool {
                                        return a.col() < b.col();
                                    });
        matrix->resize(row->row() + 1, col->col() + 1);
        matrix->setFromTriplets(_list.begin(), _list.end());
        return matrix;
    }

    template<typename T>
    std::vector<int> NonZeroCols(const std::vector<Eigen::Triplet<T>> &matrix) {
        std::vector<int> col;
        std::set<int> colIndex;
        for (const auto &i:matrix) {
            if (i.value() != 0) {
                colIndex.insert(i.col());
            }
        }
        col.resize(colIndex.size());
        std::copy(colIndex.begin(), colIndex.end(), col.begin());
        return col;
    }

    template<typename T>
    std::vector<int> NonZeroRows(const std::vector<Eigen::Triplet<T>> &matrix) {
        std::vector<int> row;
        std::set<int> rowIndex;
        for (const auto &i:matrix) {
            if (i.value() != 0) {
                rowIndex.insert(i.row());
            }
        }
        row.resize(rowIndex.size());
        std::copy(rowIndex.begin(), rowIndex.end(), row.begin());
        return row;
    }

    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>> SparseTransform(const std::vector<int> &mapInform, const int &col) {
        std::unique_ptr<Eigen::SparseMatrix<T>> matrix(new Eigen::SparseMatrix<T>);
        int row = mapInform.size();
        ASSERT(mapInform[mapInform.size() - 1] + 1<=col,"Invalide matrix col/row.");
        matrix->resize(row, col);
        for (int i = 0; i != row; i++) {
            matrix->coeffRef(i, mapInform[i]) = 1;
        }
        return matrix;
    }

    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>>
    SparseMatrixGivenColRow(const std::vector<int> &row, const std::vector<int> &col,
                            const Eigen::SparseMatrix<T> &original) {
        ASSERT(original.cols() > *(col.end() - 1) && original.rows() > *(row.end() - 1),
               "The original size of the given matrix is inconsistent with the give row/col");
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        auto colTransform = Accessory::SparseTransform<T>(col,original.cols());
        auto rowTransform = Accessory::SparseTransform<T>(row,original.rows());
        *result = (*rowTransform) * (original) * (*colTransform).transpose();
        return result;
    }
    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>>
    SparseMatrixGivenColRow(const std::vector<int> &row, const std::vector<int> &col,
                            const std::unique_ptr<Eigen::SparseMatrix<T>> &original) {
        return SparseMatrixGivenColRow<T>(row,col,*original);
    }
    template<typename T>
    std::unique_ptr<Eigen::SparseMatrix<T>>
    SparseMatrixGivenColRow(const std::vector<int> &row, const std::vector<int> &col,
                            const std::vector<Eigen::Triplet<T>> &_list) {
        auto original = Accessory::SparseMatrixMaker<T>(_list);
        return SparseMatrixGivenColRow<T>(row,col,std::move(original));
    }

    template<typename T>
    //Dense out all zero columns and rows.
    std::tuple<std::vector<int>, std::vector<int>, std::unique_ptr<Eigen::SparseMatrix<T>>>
    CondensedSparseMatrixMaker(const std::vector<Eigen::Triplet<T>> &_list) {
        auto col = NonZeroCols(_list);
        auto row = NonZeroRows(_list);
        std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
        result = SparseMatrixGivenColRow<T>(col, row, _list);
        return std::make_tuple(row, col, std::move(result));
    }
}
template<typename T>
class DofMapper {
public:
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;

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

    std::vector<int> CondensedIndexMap() {
        std::vector<int> res;
        for (const auto &i:_domains) {
            auto startIndex = StartingIndex(i);
            for (int inDomainIndex = 0, domainDof = _patchDof[i]; inDomainIndex != domainDof; ++inDomainIndex) {
                int copyIndex = inDomainIndex;
                if (FreeIndexInDomain(i, copyIndex)) {
                    res.push_back(startIndex + inDomainIndex);
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
        std::cout << "Free Dof of given domain are:" << it->second - itit->second.size();
        std::cout << std::endl;
    }

private:
    std::vector<DomainShared_ptr> _domains;
    std::map<DomainShared_ptr, int> _patchDof;
    std::map<DomainShared_ptr, std::set<int>> _freezeDof;

};


#endif //OO_IGA_DOFMAPPER_H
