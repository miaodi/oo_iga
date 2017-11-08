#pragma once

#include "VertexVisitor.hpp"

template <int N, typename T>
class BiharmonicVertexVisitor : public VertexVisitor<N, T>
{
  public:
    using Matrix = typename VertexVisitor<N, T>::Matrix;
    using Vector = typename VertexVisitor<N, T>::Vector;
    using DomainShared_ptr = typename VertexVisitor<N, T>::DomainShared_ptr;

  public:
    BiharmonicVertexVisitor(const DofMapper<N, T> &dof_mapper) : VertexVisitor<N, T>(dof_mapper) {}

  protected:
    void VertexConstraint(Vertex<N, T> *, Vertex<N, T> *);
};

template <int N, typename T>
void BiharmonicVertexVisitor<N, T>::VertexConstraint(Vertex<N, T> *master_vertex, Vertex<N, T> *slave_vertex)
{
    auto master_domain = master_vertex->Parent(0).lock()->Parent(0).lock()->GetDomain();
    auto slave_domain = slave_vertex->Parent(0).lock()->Parent(0).lock()->GetDomain();
    Vector phy_position = slave_vertex->GetDomain()->Position();
    Vector master_parametric, slave_parametric;
    if (!master_domain->InversePts(phy_position, master_parametric))
    {
        std::cerr << "Inverse mapping for vertex visitor error." << std::endl;
    }
    if (!slave_domain->InversePts(phy_position, slave_parametric))
    {
        std::cerr << "Inverse mapping for vertex visitor error." << std::endl;
    }
    auto master_eval = master_domain->EvalDerAllTensor(master_parametric, 1);
    auto slave_eval = slave_domain->EvalDerAllTensor(slave_parametric, 1);
    auto master_indices = master_vertex->IndicesForBiharmonic();
    auto slave_indices = slave_vertex->IndicesForBiharmonic();
    master_eval->erase(std::remove_if(master_eval->begin(), master_eval->end(), [&master_indices](decltype(*(master_eval->begin())) &val) {
                           if (std::find(master_indices->begin(), master_indices->end(), val.first) == master_indices->end())
                           {
                               return true;
                           }
                           return false;
                       }),
                       master_eval->end());
    slave_eval->erase(std::remove_if(slave_eval->begin(), slave_eval->end(), [&slave_indices](decltype(*(slave_eval->begin())) &val) {
                          if (std::find(slave_indices->begin(), slave_indices->end(), val.first) == slave_indices->end())
                          {
                              return true;
                          }
                          return false;
                      }),
                      slave_eval->end());
    Matrix master_matrix(3, master_eval->size()), slave_matrix(3, slave_eval->size());
    for (auto it = (*master_eval).begin(); it != (*master_eval).end(); ++it)
    {
        master_matrix(0, it - (*master_eval).begin()) = it->second[0];
        master_matrix(1, it - (*master_eval).begin()) = it->second[1];
        master_matrix(2, it - (*master_eval).begin()) = it->second[2];
    }
    for (auto it = (*slave_eval).begin(); it != (*slave_eval).end(); ++it)
    {
        slave_matrix(0, it - (*slave_eval).begin()) = it->second[0];
        slave_matrix(1, it - (*slave_eval).begin()) = it->second[1];
        slave_matrix(2, it - (*slave_eval).begin()) = it->second[2];
    }
    std::cout << master_matrix << std::endl
              << slave_matrix << std::endl
              << std::endl;
}