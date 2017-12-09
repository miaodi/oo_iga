#pragma once

#include "DomainVisitor.hpp"

template <int N, typename T>
class PostProcess : public DomainVisitor<2, N, T>
{
  public:
    using Knot = typename DomainVisitor<2, N, T>::Knot;
    using Quadrature = typename DomainVisitor<2, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<2, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<2, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<2, N, T>::KnotSpanlist;
    using Matrix = typename DomainVisitor<2, N, T>::Matrix;
    using Vector = typename DomainVisitor<2, N, T>::Vector;
    using LoadFunctor = std::function<std::vector<T>(const Knot &)>;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

  public:
    PostProcess(const PhyTensorBsplineBasis<2, 2, double> &solution, const LoadFunctor &analytical_solution, const LoadFunctor &analytical_stress_solution) : _solution(solution), _analyticalSolution(analytical_solution), _analyticalStressSolution(analytical_stress_solution)
    {
        T nu = 0.49999;
        T E = 1e11;
        T mu = E / 2 / (1 + nu);
        _constitutive.resize(3, 3);
        _constitutive << 1 - nu, nu, 0, nu, 1 - nu, 0, 0, 0, (1.0 - 2 * nu) / 2;
        _constitutive *= E / (1 + nu) / (1 - 2 * nu);
    }
    T L2Norm() const;
    T L2StressNorm() const;

  protected:
    void LocalAssemble(Element<2, N, T> *,
                       const QuadratureRule<T> &,
                       const KnotSpan &);

  protected:
    const LoadFunctor &_analyticalSolution;
    const LoadFunctor &_analyticalStressSolution;
    const PhyTensorBsplineBasis<2, 2, double> &_solution;
    std::vector<std::pair<T, T>> _normContainer;
    std::vector<std::pair<T, T>> _stressNormContainer;
    Matrix _constitutive;
};

template <int N, typename T>
void PostProcess<N, T>::LocalAssemble(Element<2, N, T> *g,
                                      const QuadratureRule<T> &quadrature_rule,
                                      const KnotSpan &knot_span)
{

    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto num_of_quadrature = quadrature_points.size();
    T relative{0}, denominator{0};
    T stress_relative{0}, stress_denominator{0};
    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        T x = _analyticalSolution(domain->AffineMap(quadrature_points[i].first))[0];
        T y = _analyticalSolution(domain->AffineMap(quadrature_points[i].first))[1];
        Vector approx_solution = _solution.AffineMap(quadrature_points[i].first);
        Vector approx_strain_solution1 = _solution.AffineMap(quadrature_points[i].first, {1, 0});
        Vector approx_strain_solution2 = _solution.AffineMap(quadrature_points[i].first, {0, 1});
        Vector strain(3);
        strain << approx_strain_solution1(0), approx_strain_solution2(1), approx_strain_solution2(0) + approx_strain_solution1(1);

        Vector approx_stress = _constitutive * strain;

        T sigma_xx = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[0];
        T sigma_yy = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[1];
        T sigma_xy = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[2];
        relative += quadrature_points[i].second * (pow(x - approx_solution(0), 2) + pow(y - approx_solution(1), 2)) * domain->Jacobian(quadrature_points[i].first);
        denominator += quadrature_points[i].second * (pow(x, 2) + pow(y, 2)) * domain->Jacobian(quadrature_points[i].first);
        stress_relative += quadrature_points[i].second * (pow(sigma_xx - approx_stress(0), 2) + pow(sigma_yy - approx_stress(1), 2) + pow(sigma_xy - approx_stress(2), 2)) * domain->Jacobian(quadrature_points[i].first);
        stress_denominator += quadrature_points[i].second * (pow(sigma_xx, 2) + pow(sigma_yy, 2) + pow(sigma_xy, 2)) * domain->Jacobian(quadrature_points[i].first);
    }
    std::lock_guard<std::mutex> lock(this->_mutex);
    _normContainer.push_back(std::make_pair(relative, denominator));
    _stressNormContainer.push_back(std::make_pair(stress_relative, stress_denominator));
}

template <int N, typename T>
T PostProcess<N, T>::L2Norm() const
{
    T relative{0}, denominator{0};
    for (const auto &i : _normContainer)
    {
        relative += i.first;
        denominator += i.second;
    }
    return sqrt(relative / denominator);
}

template <int N, typename T>
T PostProcess<N, T>::L2StressNorm() const
{
    T relative{0}, denominator{0};
    for (const auto &i : _stressNormContainer)
    {
        relative += i.first;
        denominator += i.second;
    }
    return sqrt(relative / denominator);
}