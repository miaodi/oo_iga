//
// Created by miaodi on 07/05/2017.
//

#ifndef OO_IGA_QUADRATURERULE_H
#define OO_IGA_QUADRATURERULE_H

#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <boost/multiprecision/gmp.hpp>

using namespace boost::multiprecision;

template <typename T>
class QuadratureRule
{
  public:
    using Coordinate = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Quadrature = std::pair<Coordinate, T>;
    using QuadList = std::vector<Quadrature>;
    using CoordinatePair = std::pair<Coordinate, Coordinate>;

  public:
    QuadratureRule() = default;

    ~QuadratureRule() = default;

    QuadratureRule(const int &num);

    void
    SetUpQuadrature(int num);

    static void
    LookupReference(int num, QuadList &quadrature);

    static void
    ComputeReference(int num, QuadList &quadrature, unsigned digits = std::numeric_limits<T>::digits);

    void MapToQuadrature(const CoordinatePair &range, QuadList &quadrature) const;

    int NumOfQuadrature() const;

    void PrintCurrentQuadrature() const;

  private:
    QuadList _quadrature;
    int _size;
};

#endif //OO_IGA_QUADRATURERULE_H
