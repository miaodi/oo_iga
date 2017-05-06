//
// Created by di miao on 2017/5/4.
//

#ifndef OO_IGA_MMPMATRIX_H
#define OO_IGA_MMPMATRIX_H


#include <vector>
#include <map>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include "KnotVector.h"
#include <cmath>

typedef std::vector<int> label;
using Eigen::Dynamic;
template<int Dim, int Change>
struct ChangeDim {
    enum {
        D = Change + Dim < 0 ? 0 : Dim + Change
    };
};
template<int Change>
struct ChangeDim<Dynamic, Change> {
    enum {
        D = Dynamic
    };
};

template<class T, int _Rows, int _Cols, int _Options = false>
class mmpMatrix : public Eigen::Matrix<T, _Rows, _Cols, _Options> {

    // Base is the dense matrix class of Eigen
    typedef Eigen::Matrix<T, _Rows, _Cols, _Options> Base;
    // Self type
    typedef mmpMatrix<T, _Rows, _Cols, _Options> Self;

    typedef typename Base::ColXpr ColXpr;

    typedef typename Base::RowXpr RowXpr;

    typedef typename Base::ConstColXpr ConstColXpr;

    typedef typename Base::ConstRowXpr ConstRowXpr;

    // Type pointing to a block of the matrix
    typedef Eigen::Block<Base> Block;

    // Type pointing to a (const) block of the matrix
    typedef Eigen::Block<const Base> constBlock;

    // Type pointing to a row of the matrix
    typedef Eigen::Block<Base, 1, _Cols, false> Row;

    // Type pointing to a (const) row of the matrix
    typedef Eigen::Block<const Base, 1, _Cols, false> constRow;

    // Type pointing to a set of successive rows of the matrix
    typedef Eigen::Block<Base, Dynamic, _Cols, false> Rows;

    // Type pointing to a a set of successive (const) rows of the matrix
    typedef Eigen::Block<const Base, Dynamic, _Cols, false> constRows;

    // Type pointing to a column of the matrix
    typedef Eigen::Block<Base, _Rows, 1, true> Column;

    // Type pointing to a (const) column of the matrix
    typedef Eigen::Block<const Base, _Rows, 1, true> constColumn;

    // Type pointing to a set of successive columns of the matrix
    typedef Eigen::Block<Base, _Rows, Dynamic, true> Columns;

    // Type pointing to a set of successive (const) columns of the matrix
    typedef Eigen::Block<const Base, _Rows, Dynamic, true> constColumns;


    // Type refering to any possible Eigen type that can be copied
    // into a gsMatrix
    typedef Eigen::Ref<Base> Ref;

    // Type refering to any (const) possible Eigen types that can be
    // copied into a gsMatrix
    typedef const Eigen::Ref<const Base> constRef;

    // type of first minor matrix: rows and cols reduced by one
    typedef mmpMatrix<T, ChangeDim<_Rows, -1>::D, ChangeDim<_Cols, -1>::D> FirstMinorMatrixType;

    // type of row minor matrix: rows reduced by one
    typedef mmpMatrix<T, ChangeDim<_Rows, -1>::D, _Cols> RowMinorMatrixType;

    // type of col minor matrix: cols reduced by one
    typedef mmpMatrix<T, _Rows, ChangeDim<_Cols, -1>::D> ColMinorMatrixType;

public:  // Solvers related to mmpMatrix
    typedef typename Eigen::EigenSolver<Base> EigenSolver;

    typedef typename Eigen::SelfAdjointEigenSolver<Base> SelfAdjEigenSolver;

    typedef typename Eigen::GeneralizedSelfAdjointEigenSolver<Base> GenSelfAdjEigenSolver;

    // Jacobi SVD using ColPivHouseholderQRPreconditioner
    typedef typename Eigen::JacobiSVD<Base> JacobiSVD;

public:

    mmpMatrix() = default;

    mmpMatrix(const Base &a);

    mmpMatrix(const label &, const label &);

    mmpMatrix(const Base &, const label &, const label &);

    // implicitly deleted in C++11
    //gsMatrix(const gsMatrix& a) : Base(a) { }

    mmpMatrix(int rows, int cols);

public:
    std::pair<int, int> dim() const { return std::make_pair(this->rows(), this->cols()); }

    /// \brief Returns the \a i-th element of the vectorization of the matrix
    T at(int i) const { return *(this->data() + i); }

    /// \brief Returns the \a i-th element of the vectorization of the matrix
    T &at(int i) { return *(this->data() + i); }

    ColXpr col(int i) {
        auto it = std::find(_col.begin(), _col.end(), i);
        ASSERT(it != _col.end(), "Invalid input.");
        return ColXpr(Base::derived(), it - _col.begin());
    }

    RowXpr row(int i) {
        auto it = std::find(_row.begin(), _row.end(), i);
        ASSERT(it != _row.end(), "Invalid input.");
        return RowXpr(Base::derived(), it - _row.begin());
    }

    ConstColXpr col(int i) const {
        auto it = std::find(_col.begin(), _col.end(), i);
        ASSERT(it != _col.end(), "Invalid input.");
        return ConstColXpr(Base::derived(), it - _col.begin());
    }

    ConstRowXpr row(int i) const {
        auto it = std::find(_row.begin(), _row.end(), i);
        ASSERT(it != _row.end(), "Invalid input.");
        return ConstRowXpr(Base::derived(), it - _row.begin());
    }

    void removeZeroRow() {
        Eigen::Matrix<bool, Dynamic, 1> empty = (this->array() == 0).rowwise().all();
        int num = 0;
        for (int i = 0; i < empty.size(); ++i)
            if (empty(i) == true) num++;
        Eigen::Matrix<T, Dynamic, Dynamic> tmp(this->rows() - num, this->cols());
        label row_tmp;
        row_tmp.reserve(this->rows() - num);
        int j = 0;
        for (int i = 0; i < empty.size(); ++i)
            if (empty(i) != true) {
                tmp.row(j) = this->row(i);
                j++;
                row_tmp.push_back(_row[i]);
            }
        _row=row_tmp;
        *this=tmp;
    }

    void removeZeroCol() {
        Eigen::Matrix<bool, 1, Dynamic> empty = (this->array() == 0).colwise().all();
        int num = 0;
        for (int i = 0; i < empty.size(); ++i)
            if (empty(i) == true) num++;
        Eigen::Matrix<T, Dynamic, Dynamic> tmp(this->rows(), this->cols()-num);
        label col_tmp;
        col_tmp.reserve(this->cols() - num);
        int j = 0;
        for (int i = 0; i < empty.size(); ++i)
            if (empty(i) != true) {
                tmp.col(j) = this->col(i);
                j++;
                col_tmp.push_back(_col[i]);
            }
        _col=col_tmp;
        *this=tmp;
    }

    void removeZero(){
        removeZeroCol();
        removeZeroRow();
    }
    void removeNoise(const T tol) {
        this->noalias() = this->unaryExpr(removeNoise_helper(tol));
    }

protected:
    label _row;
    label _col;

    struct removeNoise_helper {
        removeNoise_helper(const T &tol)
                : m_tol(tol) {}

        inline const T operator()(const T &val) const { return (abs(val) < m_tol ? 0 : val); }

        const T &m_tol;
    };
};

template<class T, int _Rows, int _Cols, int _Options>
inline
mmpMatrix<T, _Rows, _Cols, _Options>::mmpMatrix(const Base &a) : Base(a) {
    int i = a.rows(), j = a.cols();
    _row.resize(i);
    _col.resize(j);
    std::iota(std::begin(_row), std::end(_row), 0);
    std::iota(std::begin(_col), std::end(_col), 0);
}

template<class T, int _Rows, int _Cols, int _Options>
inline
mmpMatrix<T, _Rows, _Cols, _Options>::mmpMatrix(int rows, int cols) : Base(rows, cols), _row(rows), _col(cols) {
    std::iota(std::begin(_row), std::end(_row), 0);
    std::iota(std::begin(_col), std::end(_col), 0);
    this->setZero();
}

template<class T, int _Rows, int _Cols, int _Options>
inline
mmpMatrix<T, _Rows, _Cols, _Options>::mmpMatrix(const label &row, const label &col):Base(row.size(), col.size()),
                                                                                    _row(row), _col(col) {
    this->setZero();
}

template<class T, int _Rows, int _Cols, int _Options>
inline
mmpMatrix<T, _Rows, _Cols, _Options>::mmpMatrix(const mmpMatrix::Base &a, const label &row, const label &col):Base(a), _row(row), _col(col) {

}

#endif //OO_IGA_MMPMATRIX_H
