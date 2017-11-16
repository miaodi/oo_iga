#pragma once

#include "VertexVisitor.hpp"

template <int N, typename T>
class PoissonVertexVisitor : public VertexVisitor<N, T>
{
public:
  using Matrix = typename VertexVisitor<N, T>::Matrix;
  using Vector = typename VertexVisitor<N, T>::Vector;
  using DomainShared_ptr = typename VertexVisitor<N, T>::DomainShared_ptr;

public:
  PoissonVertexVisitor(const DofMapper<N, T> &dof_mapper) : VertexVisitor<N, T>(dof_mapper) {}

protected:
  void VertexConstraint(Vertex<N, T> *, Vertex<N, T> *);
};

template <int N, typename T>
void PoissonVertexVisitor<N, T>::VertexConstraint(Vertex<N, T> *master_vertex, Vertex<N, T> *slave_vertex)
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
  this->_constraint.push_back(Eigen::Triplet<T>(this->_dofMapper.GlobalVertexIndicesGetter(slave_vertex)[0], this->_dofMapper.GlobalVertexIndicesGetter(master_vertex)[0], 1));
}