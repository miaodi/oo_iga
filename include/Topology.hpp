//
// Created by miaodi on 10/05/2017.
//

#pragma once

#include "PhyTensorBsplineBasis.h"
#include <eigen3/Eigen/Sparse>
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

    virtual std::unique_ptr<std::vector<int>> Indices(const int &) const = 0;

    virtual void PrintIndices(const int &layerNum) const
    {
        auto res = Indices(layerNum);
        for (const auto &i : *res)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    inline DomainShared_ptr GetDomain() const
    {
        return _domain;
    }

  protected:
    DomainShared_ptr _domain;
};

template <int d, int N, typename T>
Element<d, N, T>::Element() : _domain{std::make_shared<PhyTensorBsplineBasis<d, N, T>>()}
{
}

template <int d, int N, typename T>
Element<d, N, T>::Element(const Element::DomainShared_ptr &m) : _domain{m}
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
