#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Surface.hpp"
#include "Utility.hpp"
#include "PhyTensorNURBSBasis.h"
#include "Elasticity2DDeviatoricStiffnessVisitor.hpp"
#include "PressureProjectionVisitor.hpp"
#include "PressureStiffnessVisitor.hpp"
#include "H1DomainSemiNormVisitor.hpp"
#include "NeumannBoundaryVisitor.hpp"
#include <fstream>
#include <time.h>
#include <boost/multiprecision/gmp.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::multiprecision;
using GeometryVector = PhyTensorBsplineBasis<2, 2, double>::GeometryVector;
using WeightVector = PhyTensorNURBSBasis<2, 2, double>::WeightVector;
using Vector2mpf = Matrix<mpf_float_100, 2, 1>;
using VectorXmpf = Matrix<mpf_float_100, Dynamic, 1>;
using MatrixXmpf = Matrix<mpf_float_100, Dynamic, Dynamic>;
using Vector1d = Matrix<double, 1, 1>;
void DersBasisFuns(int i, double u, int p, double *U, int n, double **&ders)
{ //checked
    ders = new double *[n + 1];
    for (int m = 0; m < n + 1; m++)
        ders[m] = new double[p + 1];
    double **ndu;
    ndu = new double *[p + 1];
    for (int m = 0; m < p + 1; m++)
        ndu[m] = new double[p + 1];
    double **a;
    a = new double *[2];
    for (int m = 0; m < 2; m++)
        a[m] = new double[p + 1];
    ndu[0][0] = 1;
    double *left;
    left = new double[p + 1];
    double *right;
    right = new double[p + 1];
    for (int j = 1; j <= p; j++)
    {
        left[j] = u - U[i + 1 - j];
        right[j] = U[i + j] - u;
        double saved = 0;
        for (int r = 0; r < j; r++)
        {
            ndu[j][r] = right[r + 1] + left[j - r];
            double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= p; j++)
        ders[0][j] = ndu[j][p];
    for (int r = 0; r <= p; r++)
    {
        int s1 = 0, s2 = 1;
        a[0][0] = 1;
        for (int k = 1; k <= n; k++)
        {
            double d = 0;
            int rk = r - k, pk = p - k;
            if (r >= k)
            {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            int j1, j2;
            if (rk >= -1)
                j1 = 1;
            else
                j1 = -rk;
            if (r - 1 <= pk)
                j2 = k - 1;
            else
                j2 = p - r;
            for (int j = j1; j <= j2; j++)
            {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk)
            {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            int j = s1;
            s1 = s2;
            s2 = j;
        }
    }
    int r = p;
    for (int k = 1; k <= n; k++)
    {
        for (int j = 0; j <= p; j++)
            ders[k][j] *= r;
        r *= (p - k);
    }
    delete[] left;
    delete[] right;
    for (int k = 0; k < 2; k++)
        delete a[k];
    delete[] a;
    for (int k = 0; k < p + 1; k++)
        delete ndu[k];
    delete[] ndu;
}
int Findspan(int m, int p, double *U, double u)
{ //checked
    /* This function determines the knot span.*/
    int n = m - p - 1;
    if (u >= U[n + 1])
        return n;
    if (u <= U[p])
        return p;
    int low = p, high = n + 1;
    int mid = (low + high) / 2;
    while (u < U[mid] || u >= U[mid + 1])
    {
        if (u < U[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
    /*Test knot={ 0,0,0,1,2,4,4,5,6,6,6 }, u=0 return 2, 2.5 return 4, 4 return 6,3.9999 return 4,
		6 return 7.
		*/
}
void Geometry(double xi, double eta, double &pxpxi, double &pxpeta, double &pypxi, double &pypeta)
{
    double knot_x[] = {0, 0, 0, 1, 1, 1};
    double knot_y[] = {0, 0, 0, 1, 1, 1};
    MatrixXd B_x(3, 3);
    MatrixXd B_y(3, 3);
    MatrixXd weights(3, 3);
    weights << 1, 1.0 / sqrt(2), 1, 1, 1.0 / sqrt(2), 1, 1, 1.0 / sqrt(2), 1;
    B_x << -1, -1, 0, -2.5, -2.5, 0, -4, -4, 0;
    B_y << 0, 1, 1, 0, 2.5, 2.5, 0, 4, 4;
    int p_x = 2, p_y = 2;
    int m_x = 5, m_y = 5;
    int dof_x = m_x - p_x, dof_y = m_y - p_y;
    int dof = dof_x * dof_y;
    MatrixXd B_xw, B_yw;
    B_xw = B_x.cwiseProduct(weights);
    B_yw = B_y.cwiseProduct(weights);
    int i_x = Findspan(m_x, p_x, knot_x, xi);
    int i_y = Findspan(m_y, p_y, knot_y, eta);
    double **ders_x, **ders_y;
    DersBasisFuns(i_x, xi, p_x, knot_x, 1, ders_x);
    DersBasisFuns(i_y, eta, p_y, knot_y, 1, ders_y);
    SparseVector<double> Nxi(dof_x), Nxi_xi(dof_x), Neta(dof_y), Neta_eta(dof_y);
    for (int kk_x = 0; kk_x < p_x + 1; kk_x++)
    {
        Nxi.insert(i_x - p_x + kk_x) = ders_x[0][kk_x];
        Nxi_xi.insert(i_x - p_x + kk_x) = ders_x[1][kk_x];
    }
    for (int kk_y = 0; kk_y < p_y + 1; kk_y++)
    {
        Neta.insert(i_y - p_y + kk_y) = ders_y[0][kk_y];
        Neta_eta.insert(i_y - p_y + kk_y) = ders_y[1][kk_y];
    }
    for (int k = 0; k < 2; k++)
        delete ders_x[k];
    delete[] ders_x;
    for (int k = 0; k < 2; k++)
        delete ders_y[k];
    delete[] ders_y;
    MatrixXd w, w_x, w_y;
    w = Neta.transpose() * weights * Nxi;
    w_x = Neta.transpose() * weights * Nxi_xi;
    w_y = Neta_eta.transpose() * weights * Nxi;
    MatrixXd pxpxi_temp, pxpeta_temp, pypxi_temp, pypeta_temp;
    pxpxi_temp = (Neta.transpose() * B_xw * Nxi_xi - Neta.transpose() * B_xw * Nxi * w_x(0, 0) / w(0, 0)) / w(0, 0);
    pypxi_temp = (Neta.transpose() * B_yw * Nxi_xi - Neta.transpose() * B_yw * Nxi * w_x(0, 0) / w(0, 0)) / w(0, 0);
    pxpeta_temp = (Neta_eta.transpose() * B_xw * Nxi - Neta.transpose() * B_xw * Nxi * w_y(0, 0) / w(0, 0)) / w(0, 0);
    pypeta_temp = (Neta_eta.transpose() * B_yw * Nxi - Neta.transpose() * B_yw * Nxi * w_y(0, 0) / w(0, 0)) / w(0, 0);
    pxpxi = pxpxi_temp(0, 0);
    pxpeta = pxpeta_temp(0, 0);
    pypxi = pypxi_temp(0, 0);
    pypeta = pypeta_temp(0, 0);
}

int main()
{
    double nu = 0.49999;
    double E = 1e11;
    double lambda = nu * E / (1 + nu) / (1 - 2 * nu);
    KnotVector<double> a;
    a.InitClosed(2, 0, 1);

    Vector2d point1(-1, 0), point2(-2.5, 0), point3(-4, 0), point4(-1, 1), point5(-2.5, 2.5), point6(-4, 4), point7(0, 1), point8(0, 2.5), point9(0, 4);
    GeometryVector points{point1, point2, point3, point4, point5, point6, point7, point8, point9};
    Vector1d weight1(1), weight2(1.0 / sqrt(2.0)), weight3(1);
    WeightVector weights{weight1, weight1, weight1, weight2, weight2, weight2, weight1, weight1, weight1};
    auto domain = make_shared<PhyTensorNURBSBasis<2, 2, double>>(std::vector<KnotVector<double>>{a, a}, points, weights, false);
    domain->DegreeElevate(2);
    domain->UniformRefine(3);
    auto cell = make_shared<Surface<2, double>>(domain);
    cell->SurfaceInitialize();

    function<vector<double>(const VectorXd &)> body_force = [](const VectorXd &u) {
        return vector<double>{0, 0};
    };
    const double pi = 3.14159265358979323846264338327;
    function<vector<double>(const VectorXd &)> stress_solution = [&pi](const VectorXd &u) {
        double x = u(0);
        double y = u(1);
        double r = sqrt(x * x + y * y);
        double theta = acos(x / r);
        double T = 1e7;
        double sigma_rr, sigma_tt, sigma_rt;
        sigma_rr = T / 2.0 * (1 - pow(1.0 / r, 2)) + T / 2.0 * (1 - 4 * pow(1.0 / r, 2) + 3 * pow(1.0 / r, 4)) * cos(2 * theta);
        sigma_tt = T / 2.0 * (1 + pow(1.0 / r, 2)) - T / 2.0 * (1 + 3 * pow(1.0 / r, 4)) * cos(2 * theta);
        sigma_rt = -T / 2.0 * (1 + 2 * pow(1.0 / r, 2) - 3 * pow(1.0 / r, 4)) * sin(2 * theta);
        MatrixXd stress_tensor_polar(2, 2), stress_tensor_cartisan(2, 2), transform(2, 2);
        transform << cos(theta), -sin(theta), sin(theta), cos(theta);
        stress_tensor_polar << sigma_rr, sigma_rt, sigma_rt, sigma_tt;
        stress_tensor_cartisan = transform * stress_tensor_polar * transform.transpose();
        return vector<double>{stress_tensor_cartisan(0, 0), stress_tensor_cartisan(1, 1), stress_tensor_cartisan(0, 1)};
    };

    Elasticity2DDeviatoricStiffnessVisitor<double> stiffness(body_force);
    PressureProjectionVisitor<double> projection;
    PressureStiffnessVisitor<double> pressure;
    H1DomainSemiNormVisitor<double> h1_norm;
    NeumannBoundaryVisitor<double> neumann(stress_solution);
    cell->Accept(stiffness);
    cell->Accept(projection);
    cell->Accept(pressure);
    cell->Accept(h1_norm);
    cell->EdgePointerGetter(2)->Accept(neumann);
    SparseMatrix<double> sparse_stiffness_triangle_view, sparse_projection, sparse_pressure, sparse_h1, rhs;
    stiffness.StiffnessAssembler(sparse_stiffness_triangle_view);
    projection.InnerProductAssembler(sparse_projection);
    pressure.InnerProductAssembler(sparse_pressure);
    h1_norm.InnerProductAssembler(sparse_h1);
    neumann.NeumannBoundaryAssembler(rhs);

    auto east_indices = cell->EdgePointerGetter(1)->Indices(0);
    auto west_indices = cell->EdgePointerGetter(3)->Indices(0);
    vector<int> dirichlet_indices;
    for (const auto &i : *east_indices)
    {
        dirichlet_indices.push_back(2 * i);
    }
    for (const auto &i : *west_indices)
    {
        dirichlet_indices.push_back(2 * i + 1);
    }
    sort(dirichlet_indices.begin(), dirichlet_indices.end());
    MatrixXd global_to_free = MatrixXd::Identity(2 * (domain->GetDof()), 2 * (domain->GetDof()));
    for (auto it = dirichlet_indices.rbegin(); it != dirichlet_indices.rend(); ++it)
    {
        Accessory::removeRow(global_to_free, *it);
    }
    SparseMatrix<double> sparse_stiffness = sparse_stiffness_triangle_view.template selfadjointView<Eigen::Upper>();
    MatrixXd stiffness_matrix = global_to_free * (lambda * sparse_projection.transpose() * sparse_pressure * sparse_projection+sparse_stiffness) * global_to_free.transpose();
    VectorXd load_vector = global_to_free * rhs;
    cout << stiffness_matrix.partialPivLu().solve(load_vector) << endl;
    // MatrixXd b_solve = global_to_free * sparse_projection.transpose() * sparse_pressure * sparse_projection * global_to_free.transpose();
    // MatrixXd solution = h1_solve.partialPivLu().solve(b_solve);
    // VectorXd eigen_value = solution.eigenvalues().array().abs();
    // sort(eigen_value.data(), eigen_value.data() + eigen_value.size());
    // cout << eigen_value.transpose() << endl;

    return 0;
}