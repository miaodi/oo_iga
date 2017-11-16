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
    auto master_eval = master_domain->Eval2PhyDerAllTensor(master_parametric);
    auto slave_eval = slave_domain->Eval2PhyDerAllTensor(slave_parametric);
    Matrix vertex_constraint(6, this->_dofMapper.Dof());
    vertex_constraint.setZero();
    int start_index_master = this->_dofMapper.StartingIndex(master_domain);
    for (auto it = (*master_eval).begin(); it != (*master_eval).end(); ++it)
    {
        vertex_constraint(0, start_index_master + it->first) = it->second[0];
        vertex_constraint(1, start_index_master + it->first) = it->second[1];
        vertex_constraint(2, start_index_master + it->first) = it->second[2];
        vertex_constraint(3, start_index_master + it->first) = it->second[3];
        vertex_constraint(4, start_index_master + it->first) = it->second[4];
        vertex_constraint(5, start_index_master + it->first) = it->second[5];
    }
    int start_index_slave = this->_dofMapper.StartingIndex(slave_domain);
    for (auto it = (*slave_eval).begin(); it != (*slave_eval).end(); ++it)
    {
        vertex_constraint(0, start_index_slave + it->first) = -it->second[0];
        vertex_constraint(1, start_index_slave + it->first) = -it->second[1];
        vertex_constraint(2, start_index_slave + it->first) = -it->second[2];
        vertex_constraint(3, start_index_slave + it->first) = -it->second[3];
        vertex_constraint(4, start_index_slave + it->first) = -it->second[4];
        vertex_constraint(5, start_index_slave + it->first) = -it->second[5];
    }
    this->_constraint.push_back(vertex_constraint);
}