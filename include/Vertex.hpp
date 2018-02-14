//
// Created by di miao on 10/9/17.
//

#ifndef OO_IGA_VERTEX_HPP
#define OO_IGA_VERTEX_HPP

#include "Topology.hpp"

template <int d, int N, typename T>
class Visitor;

template <int N, typename T>
class Vertex : public Element<0, N, T>, public std::enable_shared_from_this<Vertex<N, T>>
{
  public:
    typedef typename Element<0, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<0, N, T>::Coordinate Coordinate;
    typedef typename Element<0, N, T>::PhyPts PhyPts;
    typedef typename Element<0, N, T>::CoordinatePairList CoordinatePairList;

    Vertex() : Element<0, N, T>() {}

    Vertex(const PhyPts &point) : Element<0, N, T>(), _position(point) {}

    T Measure() const
    {
        return 0;
    }

    std::unique_ptr<std::vector<int>> Indices(const int &layer) const
    {
        auto res = _parents[0].lock()->Indices(layer);
        for (int i = 1; i < _parents.size(); ++i)
        {
            auto temp = _parents[i].lock()->Indices(layer);
            std::vector<int> intersection;
            std::set_intersection(res->begin(), res->end(), temp->begin(), temp->end(),
                                  std::back_inserter(intersection));
            *res = intersection;
        }
        return res;
    };

    std::unique_ptr<std::vector<int>> ExclusiveIndices(const int &layer) const
    {
        return Indices(layer);
    }

    void PrintIndices(const int &layerNum) const
    {
        std::cout << "Activated Dofs on this vertex are: ";
        Element<0, N, T>::PrintIndices(layerNum);
    }

    void PrintExclusiveIndices(const int &layerNum) const
    {
        std::cout << "Activated exclusive Dofs on this vertex are: ";
        Element<0, N, T>::PrintExclusiveIndices(layerNum);
    }

    void ParentSetter(const std::shared_ptr<Edge<N, T>> &parent)
    {
        _parents.push_back(std::weak_ptr<Edge<N, T>>{parent});
    }

    PhyPts Position() const
    {
        return _position;
    }

    void
        Accept(Visitor<0, N, T> &visitor)
    {
        visitor.Visit(this);
    };

  protected:
    PhyPts _position;
    std::vector<std::weak_ptr<Edge<N, T>>> _parents;
};

#endif //OO_IGA_VERTEX_HPP