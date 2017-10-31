//
// Created by di miao on 2017/6/9.
//

#ifndef OO_IGA_DOFMAPPER_H
#define OO_IGA_DOFMAPPER_H

#include "Visitor.hpp"
#include <set>

template <int d, int N, typename T>
class Element;

template <int N, typename T>
class Edge;

template <int d, int N, typename T>
class Visitor;

namespace Accessory
{

//    Return a sparse matrix of the given list. The size of the sparse matrix is the same as the maximum sive of the list
template <typename T>
std::unique_ptr<Eigen::SparseMatrix<T>>
SparseMatrixMaker(const std::vector<Eigen::Triplet<T>> &_list)
{
    using IndexedValue = Eigen::Triplet<T>;
    using IndexedValueList = std::vector<IndexedValue>;
    std::unique_ptr<Eigen::SparseMatrix<T>> matrix(new Eigen::SparseMatrix<T>);

    // Find maximum row
    auto row = std::max_element(_list.begin(),
                                _list.end(),
                                [](const IndexedValue &a,
                                   const IndexedValue &b) -> bool {
                                    return a.row() < b.row();
                                });

    // Find maximum column
    auto col = std::max_element(_list.begin(),
                                _list.end(),
                                [](const IndexedValue &a,
                                   const IndexedValue &b) -> bool {
                                    return a.col() < b.col();
                                });
    matrix->resize(row->row() + 1,
                   col->col() + 1);
    matrix->setFromTriplets(_list.begin(),
                            _list.end());
    return matrix;
}

//    Return a given size sparse matrix of the given list.
template <typename T>
std::unique_ptr<Eigen::SparseMatrix<T>>
SparseMatrixMaker(const std::vector<Eigen::Triplet<T>> &_list,
                  int row,
                  int col)
{
    using IndexedValue = Eigen::Triplet<T>;
    using IndexedValueList = std::vector<IndexedValue>;
    std::unique_ptr<Eigen::SparseMatrix<T>> matrix(new Eigen::SparseMatrix<T>);

    // Find maximum row
    auto list_row = std::max_element(_list.begin(),
                                     _list.end(),
                                     [](const IndexedValue &a,
                                        const IndexedValue &b) -> bool {
                                         return a.row() < b.row();
                                     });

    // Find maximum column
    auto list_col = std::max_element(_list.begin(),
                                     _list.end(),
                                     [](const IndexedValue &a,
                                        const IndexedValue &b) -> bool {
                                         return a.col() < b.col();
                                     });
    ASSERT(list_row <= row && list_col <= col,
           "Invalid sparse matrix size.");
    matrix->resize(row,
                   col);
    matrix->setFromTriplets(_list.begin(),
                            _list.end());
    return matrix;
}

template <typename T>
std::vector<int>
NonZeroCols(const std::vector<Eigen::Triplet<T>> &matrix)
{
    std::vector<int> col;
    std::set<int> colIndex;
    for (const auto &i : matrix)
    {
        if (i.value() != 0)
        {
            colIndex.insert(i.col());
        }
    }
    col.resize(colIndex.size());
    std::copy(colIndex.begin(),
              colIndex.end(),
              col.begin());
    return col;
}

template <typename T>
std::vector<int>
NonZeroRows(const std::vector<Eigen::Triplet<T>> &matrix)
{
    std::vector<int> row;
    std::set<int> rowIndex;
    for (const auto &i : matrix)
    {
        if (i.value() != 0)
        {
            rowIndex.insert(i.row());
        }
    }
    row.resize(rowIndex.size());
    std::copy(rowIndex.begin(),
              rowIndex.end(),
              row.begin());
    return row;
}

template <typename T>
void SparseTransform(const std::vector<int> &map_info,
                     const int &original_dof,
                     Eigen::SparseMatrix<T> &sparse_matrix)
{
    int row = map_info.size();
    int col = original_dof;
    sparse_matrix.resize(row, col);
    for (int i = 0; i != row; i++)
    {
        sparse_matrix.coeffRef(i, map_info[i]) = 1;
    }
}

template <typename T>
std::unique_ptr<Eigen::SparseMatrix<T>>
SparseMatrixGivenColRow(const std::vector<int> &row,
                        const std::vector<int> &col,
                        const Eigen::SparseMatrix<T> &original)
{
    ASSERT(original.cols() > *(col.end() - 1) && original.rows() > *(row.end() - 1),
           "The original size of the given matrix is inconsistent with the give row/col");
    std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
    auto colTransform = Accessory::SparseTransform<T>(col,
                                                      original.cols());
    auto rowTransform = Accessory::SparseTransform<T>(row,
                                                      original.rows());
    *result = (*rowTransform) * (original) * (*colTransform).transpose();
    return result;
}

template <typename T>
std::unique_ptr<Eigen::SparseMatrix<T>>
SparseMatrixGivenColRow(const std::vector<int> &row,
                        const std::vector<int> &col,
                        const std::unique_ptr<Eigen::SparseMatrix<T>> &original)
{
    return SparseMatrixGivenColRow<T>(row,
                                      col,
                                      *original);
}

template <typename T>
std::unique_ptr<Eigen::SparseMatrix<T>>
SparseMatrixGivenColRow(const std::vector<int> &row,
                        const std::vector<int> &col,
                        const std::vector<Eigen::Triplet<T>> &_list)
{
    auto original = Accessory::SparseMatrixMaker<T>(_list);
    return SparseMatrixGivenColRow<T>(row,
                                      col,
                                      std::move(original));
}

//Dense out all zero columns and rows.
template <typename T>
std::tuple<std::vector<int>, std::vector<int>, std::unique_ptr<Eigen::SparseMatrix<T>>>
CondensedSparseMatrixMaker(const std::vector<Eigen::Triplet<T>> &_list)
{
    auto col = NonZeroCols(_list);
    auto row = NonZeroRows(_list);
    std::unique_ptr<Eigen::SparseMatrix<T>> result(new Eigen::SparseMatrix<T>);
    result = SparseMatrixGivenColRow<T>(col,
                                        row,
                                        _list);
    return std::make_tuple(row,
                           col,
                           std::move(result));
}

template <typename T>
void removeRow(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix,
               unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (rowToRemove < numRows)
        matrix.block(rowToRemove,
                     0,
                     numRows - rowToRemove,
                     numCols) = matrix.block(rowToRemove + 1,
                                             0,
                                             numRows - rowToRemove,
                                             numCols);

    matrix.conservativeResize(numRows,
                              numCols);
}

template <typename T>
void removeColumn(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix,
                  unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (colToRemove < numCols)
        matrix.block(0,
                     colToRemove,
                     numRows,
                     numCols - colToRemove) = matrix.block(0,
                                                           colToRemove + 1,
                                                           numRows,
                                                           numCols - colToRemove);

    matrix.conservativeResize(numRows,
                              numCols);
}
}

template <int N, typename T>
class DofMapper
{
  public:
    using DomainShared_ptr = typename Element<2, N, T>::DomainShared_ptr;

  public:
    DofMapper(){};

    //insert a domain pointer into the container.
    void
    DomainLabel(const DomainShared_ptr &domain)
    {
        if (std::find(_domains.begin(),
                      _domains.end(),
                      domain) == _domains.end())
        {
            _domains.push_back(domain);
            _DirichletDof[domain] = std::set<int>{};
            _slaveDof[domain] = std::set<int>{};
        }
        else
        {
            // Do nothing.
        }
    }

    // Return the index of given domain smart ptr.
    int
    DomainIndex(const DomainShared_ptr &domain) const
    {
        int res = std::find(_domains.begin(),
                            _domains.end(),
                            domain) -
                  _domains.begin();
        ASSERT(res < _domains.size(),
               "unknown domain. You should insert it first.");
        return res;
    }

    // Set the dof of given domain.
    void
    PatchDofSetter(const DomainShared_ptr &domain,
                   const int &num)
    {
        ASSERT(_patchDof.find(domain) == _patchDof.end(),
               "Already initialized");
        _patchDof[domain] = num;
    }

    // Set the Dirichlet index for the given domain
    void
    DirichletDofInserter(const DomainShared_ptr &domain,
                         const int &num)
    {
        _DirichletDof[domain].insert(num);
    }

    // Set the slave index for the given domain
    void
    SlaveDofInserter(const DomainShared_ptr &domain,
                     const int &num)
    {
        _slaveDof[domain].insert(num);
    }

    void
    EdgeIndicesInserter(Edge<N, T> *g,
                        const std::vector<int> &indices)
    {
        for (const auto &i : indices)
        {
            _edgeDof[g].insert(i);
        }
    }

    std::vector<int>
    GlobalEdgeIndicesGetter(Edge<N, T> *g) const
    {
        auto it = _edgeDof.find(g);
        std::vector<int> res(it->second.begin(), it->second.end());
        IndicesToGlobal(g->Parent(0).lock()->GetDomain(), res);
        return res;
    }

    // Return the start global index of given domain
    int
    StartingIndex(const DomainShared_ptr &domain) const
    {
        int index = DomainIndex(domain);
        int res = 0;
        for (int i = 0; i < index; ++i)
        {
            res += _patchDof.at(_domains[i]);
        }
        return res;
    }

    // Return the condensed starting index of given domain (After condense out the slave indices)
    int
    CondensedStartingIndex(const DomainShared_ptr &domain) const
    {
        int res = StartingIndex(domain);
        for (auto it = _domains.begin(); *it != domain; it++)
        {
            auto slave_indices = _slaveDof.find(*it);
            if (slave_indices != _slaveDof.end())
            {
                res -= slave_indices->second.size();
            }
        }
        return res;
    }

    // Return the unfreezed starting index of given domain (consider both Dirichlet boundary and slave d.o.f)
    int
    FreeStartingIndex(const DomainShared_ptr &domain) const
    {
        int res = StartingIndex(domain);
        for (auto it = _domains.begin(); *it != domain; it++)
        {
            auto slave_indices = _slaveDof.find(*it);
            auto Dirichlet_indices = _DirichletDof.find(*it);
            if (slave_indices != _slaveDof.end())
            {
                res -= slave_indices->second.size();
            }
            if (Dirichlet_indices != _DirichletDof.end())
            {
                res -= Dirichlet_indices->second.size();
            }
        }
        return res;
    }

    // Return the global dof of the entire problem
    int
    Dof() const
    {
        int res = 0;
        for (const auto &i : _domains)
        {
            res += _patchDof.at(i);
        }
        return res;
    }

    // Return the global dof after condense out slave
    int
    CondensedDof() const
    {
        int res = 0;
        for (const auto &i : _domains)
        {
            res += _patchDof.at(i);
            auto slave_indices = _slaveDof.find(i);
            if (slave_indices != _slaveDof.end())
            {
                res -= slave_indices->second.size();
            }
        }
        return res;
    }

    // Return the global dof after condense out slave and Dirichlet boundary
    int
    FreeDof() const
    {
        int res = 0;
        for (const auto &i : _domains)
        {
            res += _patchDof.at(i);
            auto slave_indices = _slaveDof.find(i);
            auto Dirichlet_indices = _DirichletDof.find(i);
            if (slave_indices != _slaveDof.end())
            {
                res -= slave_indices->size();
            }
            if (Dirichlet_indices != _DirichletDof.end())
            {
                res -= Dirichlet_indices->size();
            }
        }
        return res;
    }

    // Return the global index of the given domain and local index.
    int
    Index(const DomainShared_ptr &domain,
          const int &i) const
    {
        return StartingIndex(domain) + i;
    }

    // Return the local index of the given domain and local index after condense out slave.
    bool
    CondensedIndexInDomain(const DomainShared_ptr &domain,
                           int &i) const
    {
        auto domainIndex = DomainIndex(domain);
        if (_patchDof.at(_domains[domainIndex]) <= i)
            return false;
        auto slave_indices = _slaveDof.find(domain);
        if (slave_indices->second.find(i) != slave_indices->second.end())
            return false;
        int count = 0;
        count += std::count_if(_slaveDof.at(_domains[domainIndex]).cbegin(),
                               _slaveDof.at(_domains[domainIndex]).cend(),
                               [&i](int num) { return num < i; });
        i -= count;
        return true;
    }

    // Return the local index of the given domain and local index after condense out slave and Dirichlet boundary.
    bool
    FreeIndexInDomain(const DomainShared_ptr &domain,
                      int &i) const
    {
        auto domainIndex = DomainIndex(domain);
        if (_patchDof.at(_domains[domainIndex]) <= i)
            return false;
        auto slave_indices = _slaveDof.find(domain);
        auto Dirichlet_indices = _DirichletDof.find(domain);
        if (slave_indices->second.find(i) != slave_indices->second.end())
            return false;
        if (Dirichlet_indices->second.find(i) != Dirichlet_indices->second.end())
            return false;
        int count = 0;
        count += std::count_if(_slaveDof.at(_domains[domainIndex]).begin(),
                               _slaveDof.at(_domains[domainIndex]).end(),
                               [&i](int num) { return num < i; });
        count += std::count_if(_DirichletDof.at(_domains[domainIndex]).begin(),
                               _DirichletDof.at(_domains[domainIndex]).end(),
                               [&i](int num) { return num < i; });
        i -= count;
        return true;
    }

    // Return the global index of the given domain and local index after condense out slave and Dirichlet boundary.
    bool
    FreeIndex(const DomainShared_ptr &domain,
              int &i) const
    {
        auto res = FreeIndexInDomain(domain,
                                     i);
        if (res)
        {
            i += FreeStartingIndex(domain);
        }
        return res;
    }

    // Return the global index of the given domain and local index after condense out slave.
    bool
    CondensedIndex(const DomainShared_ptr &domain,
                   int &i) const
    {
        auto res = CondensedIndexInDomain(domain,
                                          i);
        if (res)
        {
            i += CondensedStartingIndex(domain);
        }
        return res;
    }

    //    Return true if the given global index i is not a slave index and modify i to the condensed index.
    bool
    GlobalToCondensedIndex(int &i) const
    {
        for (auto it = _domains.cbegin(); it != _domains.cend(); ++it)
        {
            if (StartingIndex(*it) + DofIn(*it) > i)
            {
                i -= StartingIndex(*it);
                return CondensedIndex(*it,
                                      i);
            }
        }
        return false;
    }

    // Return a vector of global indices of slave d.o.f in the given domain
    std::vector<int>
    SlaveDofIn(const DomainShared_ptr &domain) const
    {
        int initial = StartingIndex(domain);
        std::vector<int> res;
        auto it = _slaveDof.find(domain);
        ASSERT(it != _slaveDof.end(),
               "Given domain does not exist in the _slaveDof list.\n");
        for (const auto &i : it->second)
        {
            res.push_back(initial + i);
        }
        return res;
    }

    // Return a vector with vector index as free index, vector element as global index
    std::vector<int>
    FreeIndexMap() const
    {
        std::vector<int> res;
        for (const auto &i : _domains)
        {
            int domainDof = _patchDof.find(i)->second;
            auto startIndex = StartingIndex(i);
            for (int inDomainIndex = 0; inDomainIndex != domainDof; ++inDomainIndex)
            {
                int copyIndex = inDomainIndex;
                if (FreeIndexInDomain(i,
                                      copyIndex))
                {
                    res.push_back(startIndex + inDomainIndex);
                }
            }
        }
        return res;
    }

    void
    FreeIndexMap(Eigen::SparseMatrix<T> &sparse_matrix)
    {
        auto free_index = FreeIndexMap();
        Accessory::SparseTransform(free_index, Dof(), sparse_matrix);
    }

    // Return a vector with vector index as condensed index, vector element as global index
    std::vector<int>
    CondensedIndexMap() const
    {
        std::vector<int> res;
        for (const auto &i : _domains)
        {
            auto startIndex = StartingIndex(i);
            int domainDof = _patchDof.find(i)->second;
            for (int inDomainIndex = 0; inDomainIndex != domainDof; ++inDomainIndex)
            {
                int copyIndex = inDomainIndex;

                // If the given local index in domain i is not a slave index, push back.
                if (CondensedIndexInDomain(i,
                                           copyIndex))
                {
                    res.push_back(startIndex + inDomainIndex);
                }
            }
        }
        return res;
    }

    void
    CondensedIndexMap(Eigen::SparseMatrix<T> &sparse_matrix)
    {
        auto free_index = CondensedIndexMap();
        Accessory::SparseTransform(free_index, Dof(), sparse_matrix);
    }

    // Return a vector with vector index as free index, vector element as condensed index
    std::vector<int>
    FreeToCondensedIndexMap() const
    {
        auto condensed_indices = CondensedIndexMap();
        auto free_indices = FreeIndexMap();
        std::vector<int> res;
        for (const auto &i : free_indices)
        {
            auto it = std::find(condensed_indices.begin(),
                                condensed_indices.end(),
                                i);
            ASSERT(it != condensed_indices.end(),
                   "Inconsistence happens between free indices and condensed indices.");
            res.push_back(it - condensed_indices.begin());
        }
        return res;
    }

    void
    FreeToCondensedIndexMap(Eigen::SparseMatrix<T> &sparse_matrix)
    {
        auto free_index = FreeToCondensedIndexMap();
        Accessory::SparseTransform(free_index, CondensedDof(), sparse_matrix);
    }

    // Print out all Dirichlet local indices in the given domain.
    void
    PrintDirichletLocalIndicesIn(const DomainShared_ptr &domain) const
    {
        auto it = _DirichletDof.find(domain);
        if (it == _DirichletDof.end())
        {
            std::cout << "No Boundary Dof is found." << std::endl;
            return;
        }
        std::cout << "Boundary Dof Index in given domain are:";
        for (const auto &i : it->second)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    // Print out all Dirichlet global indices in the given domain.
    void
    PrintDirichletGlobalIndicesIn(const DomainShared_ptr &domain) const
    {
        auto it = _DirichletDof.find(domain);
        if (it == _DirichletDof.end())
        {
            std::cout << "No Boundary Dof is found." << std::endl;
            return;
        }
        std::cout << "Boundary Dof Index in given domain are:";
        for (const auto &i : it->second)
        {
            std::cout << Index(domain,
                               i)
                      << " ";
        }
        std::cout << std::endl;
    }

    // Print out all slave local indices in the given domain.
    void
    PrintSlaveLocalIndicesIn(const DomainShared_ptr &domain) const
    {
        auto it = _slaveDof.find(domain);
        if (it->second.size() == 0)
        {
            std::cout << "No Slave Dof is found." << std::endl;
            return;
        }
        std::cout << "Slave Dof Index in given domain are:";
        for (const auto &i : it->second)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    // Print out all slave global indices in the given domain.
    void
    PrintSlaveGlobalIndicesIn(const DomainShared_ptr &domain) const
    {
        auto it = _slaveDof.find(domain);
        if (it->second.size() == 0)
        {
            std::cout << "No Slave Dof is found." << std::endl;
            return;
        }
        std::cout << "Slave Dof Index in given domain are:";
        for (const auto &i : it->second)
        {
            std::cout << Index(domain,
                               i)
                      << " ";
        }
        std::cout << std::endl;
    }

    int DofIn(const DomainShared_ptr &domain) const
    {
        auto it = _patchDof.find(domain);
        if (it != _patchDof.end())
        {
            return it->second;
        }
        else
        {
            std::cout << "unknown domain ptr." << std::endl;
            return 0;
        }
    }

    // Print out the d.o.f in the given domain.
    void
    PrintDofIn(const DomainShared_ptr &domain) const
    {
        auto it = _patchDof.find(domain);
        std::cout << "Dof of given domain are: " << it->second;
        std::cout << std::endl;
    }

    // Print out Condensed d.o.f in the given domain
    void
    PrintCondensedDofIn(const DomainShared_ptr &domain) const
    {
        auto it = _patchDof.find(domain);
        auto itit = _slaveDof.find(domain);
        std::cout << "Condensed Dof of given domain are: " << it->second - itit->second.size();
        std::cout << std::endl;
    }

    // Print out Free d.o.f in the given domain
    void
    PrintFreeDofIn(const DomainShared_ptr &domain) const
    {
        auto it = _patchDof.find(domain);
        auto slave_it = _slaveDof.find(domain);
        auto Dirichlet_it = _DirichletDof.find(domain);
        std::cout << "Free Dof of given domain are: "
                  << it->second - slave_it->second.size() - Dirichlet_it.size();
        std::cout << std::endl;
    }

    void
    IndicesToGlobal(const DomainShared_ptr &domain,
                    std::vector<int> &indices) const
    {
        int start_index = StartingIndex(domain);
        std::transform(indices.cbegin(),
                       indices.cend(),
                       indices.begin(),
                       [&start_index](const int &i) { return i + start_index; });
    }

    //    All global Dirichlet indices.
    std::vector<int>
    GlobalDirichletIndices() const
    {
        std::vector<int> res;
        for (const auto &i : _DirichletDof)
        {
            int start_index = StartingIndex(i.first);
            for (const auto &j : i.second)
            {
                res.push_back(start_index + j);
            }
        }
        return res;
    }

    //    Inverse map of GlobalDirichletIndices().
    std::map<int, int>
    GlobalDirichletCondensedMap() const
    {
        std::map<int, int> res;
        int index = 0;
        for (const auto &i : _DirichletDof)
        {
            int start_index = StartingIndex(i.first);
            for (const auto &j : i.second)
            {
                res[start_index + j] = index;
                index++;
            }
        }
        return res;
    }

    bool IsInDomain(const DomainShared_ptr &domain,
                    const int &index) const
    {
        auto it = _patchDof.find(domain);
        if (index < StartingIndex(domain) || index >= StartingIndex(domain) + (*it).second)
        {
            return false;
        }
        return true;
    }

  private:
    //! container of domain smart ptr
    std::vector<DomainShared_ptr> _domains;
    //! map domain smart ptr to d.o.f
    std::map<const DomainShared_ptr, int> _patchDof;
    //! map domain smart ptr to sorted dirichlet d.o.f
    std::map<const DomainShared_ptr, std::set<int>> _DirichletDof;
    //! map domain smart ptr to sorted slave d.o.f
    std::map<const DomainShared_ptr, std::set<int>> _slaveDof;

    std::map<Edge<N, T> *, std::set<int>> _edgeDof;
};

#endif //OO_IGA_DOFMAPPER_H
