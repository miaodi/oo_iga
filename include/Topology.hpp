//
// Created by miaodi on 10/05/2017.
//

#pragma once

#include "PhyTensorBsplineBasis.h"
#include <Eigen/Sparse>
#include "Visitor.hpp"

template <int d, int N, typename T>
class Visitor;

template <int d, int N, typename T>
class Element
{
  public:
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<d, N, T>>;

    //! Coordinate in the parametric domain.
    using Coordinate = Eigen::Matrix<T, d, 1>;
    using PhyPts = Eigen::Matrix<T, N, 1>;
    using LoadFunctor = std::function<std::vector<T>(const Coordinate &)>;
    using CoordinatePairList = typename std::vector<std::pair<Coordinate, Coordinate>>;

    Element();

    Element(const DomainShared_ptr &m);

    //! Area for surface, length for Edge, 0 for Vertex
    virtual T Measure() const = 0;

    virtual void Accept(Visitor<d, N, T> &) = 0;

    // give indices of the element for given dimension problem.
    virtual std::vector<int> Indices(const int &, const int &) const = 0;

    virtual std::vector<int> ExclusiveIndices(const int &, const int &) const = 0;

    virtual void PrintIndices(const int &dimension, const int &layerNum) const
    {
        auto res = Indices(dimension, layerNum);
        for (const auto &i : res)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    inline DomainShared_ptr GetDomain() const
    {
        return _domain;
    }

    int GetID() const
    {
        return _currentID;
    }

  protected:
    DomainShared_ptr _domain;
    static int _ID;
    const int _currentID;
};

template <int d, int N, typename T>
int Element<d, N, T>::_ID = 0;

template <int d, int N, typename T>
Element<d, N, T>::Element() : _domain{std::make_shared<PhyTensorBsplineBasis<d, N, T>>()}, _currentID(++_ID)
{
}

template <int d, int N, typename T>
Element<d, N, T>::Element(const Element::DomainShared_ptr &m) : _domain{m}, _currentID(++_ID)
{
}

// +---------------------------+
// |          north            |
// | west                 east |
// |          south            |
// +---------------------------+
enum class Orientation
{
    south,
    east,
    north,
    west
};

enum class VertexIndex
{
    first,
    second,
    third,
    fourth
};