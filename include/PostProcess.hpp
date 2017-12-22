#pragma once

#include "DomainVisitor.hpp"
#include <fstream>
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
    PostProcess(const PhyTensorBsplineBasis<2, 2, double> &solution, const PhyTensorBsplineBasis<2, 1, double> &pressure, const LoadFunctor &analytical_solution, const LoadFunctor &analytical_stress_solution) : _solution(solution), _analyticalSolution(analytical_solution), _analyticalStressSolution(analytical_stress_solution), _pressure(pressure)
    {
        T nu = 0.49999;
        T E = 1e11;
        T mu = E / 2 / (1 + nu);
        _constitutive.resize(4, 4);
        _constitutive << 1 - nu, nu, nu, 0, nu, 1 - nu, nu, 0, nu, nu, 1 - nu, 0, 0, 0, 0, (1.0 - 2 * nu) / 2;
        _constitutive *= E / (1 + nu) / (1 - 2 * nu);
        myfile_xx.open("sigma_xx.txt");
        myfile_yy.open("sigma_yy.txt");
        myfile_xy.open("sigma_xy.txt");
    }
    T L2Norm() const;
    T L2StressNorm() const;
    T L2EnergyNorm() const;
    void Plot();

  protected:
    void LocalAssemble(Element<2, N, T> *,
                       const QuadratureRule<T> &,
                       const KnotSpan &);

  protected:
    const LoadFunctor &_analyticalSolution;
    const LoadFunctor &_analyticalStressSolution;
    const PhyTensorBsplineBasis<2, 2, double> &_solution;
    const PhyTensorBsplineBasis<2, 1, double> &_pressure;
    std::vector<std::pair<T, T>> _normContainer;
    std::vector<std::pair<T, T>> _stressNormContainer;
    std::vector<std::pair<T, T>> _energyNormContainer;
    Matrix _constitutive;
    std::ofstream myfile_xx;
    std::ofstream myfile_yy;
    std::ofstream myfile_xy;
    Element<2, N, T> *_f;
};

template <int N, typename T>
void PostProcess<N, T>::LocalAssemble(Element<2, N, T> *g,
                                      const QuadratureRule<T> &quadrature_rule,
                                      const KnotSpan &knot_span)
{

    auto domain = g->GetDomain();
    _f = g;
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);
    auto num_of_quadrature = quadrature_points.size();
    T relative{0}, denominator{0};
    T stress_relative{0}, stress_denominator{0};
    T energy_relative{0}, energy_denominator{0};
    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        T x = _analyticalSolution(domain->AffineMap(quadrature_points[i].first))[0];
        T y = _analyticalSolution(domain->AffineMap(quadrature_points[i].first))[1];
        Vector approx_solution = _solution.AffineMap(quadrature_points[i].first);
        Vector approx_strain_solution1 = _solution.AffineMap(quadrature_points[i].first, {1, 0});
        Vector approx_strain_solution2 = _solution.AffineMap(quadrature_points[i].first, {0, 1});

        Vector pressure_solution = _pressure.AffineMap(quadrature_points[i].first);
        Vector u(2), v(2);
        u << approx_strain_solution1(0), approx_strain_solution2(0);
        v << approx_strain_solution1(1), approx_strain_solution2(1);
        Matrix Jacobian = domain->JacobianMatrix(quadrature_points[i].first).transpose();
        u = Jacobian.inverse() * u;
        v = Jacobian.inverse() * v;
        T volumetric = 1.0 / 3 * (u(0) + v(1));
        Vector strain(4);
        strain << u(0) - volumetric + 1.0 / 3 * pressure_solution(0), v(1) - volumetric + 1.0 / 3 * pressure_solution(0), -volumetric + 1.0 / 3 * pressure_solution(0), v(0) + u(1);

        T sigma_xx = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[0];
        T sigma_yy = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[1];
        T sigma_xy = _analyticalStressSolution(domain->AffineMap(quadrature_points[i].first))[2];

        Vector exact_stress(3);
        exact_stress << sigma_xx, sigma_yy, sigma_xy;
        Matrix transform(3, 4);
        transform.setZero();
        transform(0, 0) = 1;
        transform(1, 1) = 1;
        transform(2, 3) = 1;
        // Vector exact_strain = (transform * _constitutive * transform.transpose()).partialPivLu().solve(exact_stress);
        Vector stress = _constitutive * strain;
        Vector stress_error(3);
        stress_error << stress(0) - exact_stress(0), stress(1) - exact_stress(1), stress(3) - exact_stress(2);
        relative += quadrature_points[i].second * (pow(x - approx_solution(0), 2) + pow(y - approx_solution(1), 2)) * domain->Jacobian(quadrature_points[i].first);
        denominator += quadrature_points[i].second * (pow(x, 2) + pow(y, 2)) * domain->Jacobian(quadrature_points[i].first);
        stress_relative += quadrature_points[i].second * (pow(exact_stress(0) - stress(0), 2) + pow(exact_stress(1) - stress(1), 2) + pow(exact_stress(2) - stress(3), 2)) * domain->Jacobian(quadrature_points[i].first);
        stress_denominator += quadrature_points[i].second * (pow(exact_stress(0), 2) + pow(exact_stress(1), 2) + pow(exact_stress(2), 2)) * domain->Jacobian(quadrature_points[i].first);
        energy_relative += quadrature_points[i].second * (stress_error * (transform * _constitutive * transform.transpose()).partialPivLu().solve(stress_error))(0) * domain->Jacobian(quadrature_points[i].first);
        energy_denominator += quadrature_points[i].second * (exact_stress * (transform * _constitutive * transform.transpose()).partialPivLu().solve(exact_stress))(0) * domain->Jacobian(quadrature_points[i].first);
    }
    std::lock_guard<std::mutex> lock(this->_mutex);
    _normContainer.push_back(std::make_pair(relative, denominator));
    _stressNormContainer.push_back(std::make_pair(stress_relative, stress_denominator));
    _energyNormContainer.push_back(std::make_pair(energy_relative, energy_denominator));
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

template <int N, typename T>
T PostProcess<N, T>::L2EnergyNorm() const
{
    T relative{0}, denominator{0};
    for (const auto &i : _energyNormContainer)
    {
        relative += i.first;
        denominator += i.second;
    }
    return sqrt(relative / denominator);
}

template <int N, typename T>
void PostProcess<N, T>::Plot()
{
    auto domain = _f->GetDomain();
    Vector position(2);
    for (int i = 0; i < 101; i++)
    {
        for (int j = 0; j < 101; j++)
        {
            position << i * 1.0 / 100, j * 1.0 / 100;
            Vector approx_solution = _solution.AffineMap(position);
            Vector approx_strain_solution1 = _solution.AffineMap(position, {1, 0});
            Vector approx_strain_solution2 = _solution.AffineMap(position, {0, 1});
            T sigma_xx = _analyticalStressSolution(domain->AffineMap(position))[0];
            T sigma_yy = _analyticalStressSolution(domain->AffineMap(position))[1];
            T sigma_xy = _analyticalStressSolution(domain->AffineMap(position))[2];
            Vector pressure_solution = _pressure.AffineMap(position);
            Vector u(2), v(2);
            u << approx_strain_solution1(0), approx_strain_solution2(0);
            v << approx_strain_solution1(1), approx_strain_solution2(1);
            Matrix Jacobian = domain->JacobianMatrix(position).transpose();
            u = Jacobian.inverse() * u;
            v = Jacobian.inverse() * v;
            T volumetric = 1.0 / 3 * (u(0) + v(1));
            Vector strain(4);
            strain << u(0) - volumetric + 1.0 / 3 * pressure_solution(0), v(1) - volumetric + 1.0 / 3 * pressure_solution(0), -volumetric + 1.0 / 3 * pressure_solution(0), v(0) + u(1);
            Vector stress = _constitutive * strain;
            myfile_xx << -domain->AffineMap(position)(0) << " " << domain->AffineMap(position)(1) << " " << stress(0) << std::endl;
            myfile_yy << -domain->AffineMap(position)(0) << " " << domain->AffineMap(position)(1) << " " << stress(1) << std::endl;
            myfile_xy << -domain->AffineMap(position)(0) << " " << domain->AffineMap(position)(1) << " " << stress(3) << std::endl;
        }
    }
}