#pragma once
#include "Visitor.hpp"

template <int N, typename T>
class VertexVisitor : public Visitor<0, N, T>
{
public:
  using Matrix = typename DomainVisitor<0, N, T>::Matrix;
  using Vector = typename DomainVisitor<0, N, T>::Vector;
  using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

public:
  VertexVisitor(const DofMapper<N, T> &dof_mapper) : _dofMapper(dof_mapper) {}

  void Visit(Element<0, N, T> *);

  void
  ConstraintMatrix(Eigen::SparseMatrix<T> &);

protected:
  virtual void VertexConstraint(Vertex<N, T> *, Vertex<N, T> *) = 0;

protected:
  std::vector<Eigen::Triplet<T>> _constraint;
  const DofMapper<N, T> &_dofMapper;
};

template <int N, typename T>
void VertexVisitor<N, T>::Visit(Element<0, N, T> *g)
{
  auto vertex = dynamic_cast<Vertex<N, T> *>(g);
  if (!vertex->IsMaster() || vertex->IsDirichlet())
    return;
  auto slave_vertices = vertex->Counterpart();
  for (const auto &i : slave_vertices)
  {
    VertexConstraint(vertex, &*(i.lock()));
  }
}

template <int N, typename T>
void VertexVisitor<N, T>::ConstraintMatrix(Eigen::SparseMatrix<T> &sparse_constraint)
{
  std::vector<Eigen::Triplet<T>> constraint_triplet;
  for (int i = 0; i < this->_dofMapper.Dof(); ++i)
  {
    constraint_triplet.push_back(Eigen::Triplet<T>(i, i, 1));
  }
  if (_constraint.size())
  {
    constraint_triplet.insert(constraint_triplet.end(), _constraint.begin(), _constraint.end());
  }
  sparse_constraint.resize(this->_dofMapper.Dof(), this->_dofMapper.Dof());
  sparse_constraint.setFromTriplets(constraint_triplet.begin(), constraint_triplet.end());
}