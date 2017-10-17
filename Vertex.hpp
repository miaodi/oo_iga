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

    Vertex() : Element<0, N, T>(){};

    Vertex(const DomainShared_ptr &domain, const bool &boundary) : Element<0, N, T>(domain), _Dirichlet{boundary} {};

    Vertex(const PhyPts &point, const bool &boundary) : Element<0, N, T>(
                                                            std::make_shared<PhyTensorBsplineBasis<0, N, T>>(point)),
                                                        _Dirichlet{boundary} {};

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

    void MasterSetter(const bool &master)
    {
        _master = master;
    }

    bool IsMaster() const
    {
        return _master;
    }

    bool IsSlave() const
    {
        return !_master;
    }

    auto
    Parent(const int &i) const
    {
        return _parents[i];
    }

    void Accept(Visitor<0, N, T> &visitor)
    {
        visitor.Visit(this);
    };

    bool IsDirichlet() const
    {
        return _Dirichlet;
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

    void PrintInfo() const
    {
        std::cout << this->_domain->Position() << std::endl;
        if (_master)
        {
            std::cout << "Master vertex." << std::endl;
        }
        else
        {
            std::cout << "Slave vertex." << std::endl;
        }
        if (_Dirichlet)
        {
            std::cout << "Dirichlet vertex." << std::endl;
        }
    }

    void ParentSetter(const std::shared_ptr<Edge<N, T>> &parent)
    {
        _parents.push_back(std::weak_ptr<Edge<N, T>>{parent});
    }

    // Match two vertex from two different patch
    // (TODO) need to consider the case one of the vertices is Dirichlet
    bool Match(std::shared_ptr<Vertex<N, T>> &counterpart)
    {
        if (!_master || _Dirichlet)
        {
            return false;
        }
        if (this->_domain->Position() == counterpart->_domain->Position())
        {
            _pairList.push_back(std::weak_ptr<Vertex<N, T>>(counterpart));
            counterpart->MasterSetter(false);
            counterpart->_pairList.push_back(std::weak_ptr<Vertex<N, T>>(this->shared_from_this()));
            return true;
        }
        else
        {
            return false;
        }
    }

  protected:
    std::vector<std::weak_ptr<Vertex<N, T>>> _pairList;
    bool _Dirichlet{false};

    //All Vertices are defined as master and the match algorithm will sort such that
    //only one vertex is master for each physical vertex.
    bool _master{true};

    std::vector<std::weak_ptr<Edge<N, T>>> _parents;
};

#endif //OO_IGA_VERTEX_HPP
