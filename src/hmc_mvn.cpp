// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath> 
#include <random>
#include <functional>

#pragma GCC diagnostic pop

using namespace Rcpp;
using namespace Eigen;

// HMC bits converted from Julia 
// negative log density
double U(const VectorXd& x, VectorXd mu, VectorXd sigma) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += -std::log(sigma(i)) + 0.5 * std::log(2 * M_PI) 
               + 0.5 * std::pow((x(i) - mu(i)) / sigma(i), 2);
    }
    return sum;
}

// gradient thereof
VectorXd dU(const VectorXd& x, VectorXd mu, VectorXd sigma) {
    VectorXd grad(x.size());
    for (int i = 0; i < x.size(); ++i) {
        grad(i) = (x(i) - mu(i)) / (sigma(i) * sigma(i));
    }
    return grad;
}

VectorXd hmc_step(const VectorXd& q,
                  double epsilon,
                  int L,
                  const MatrixXd& M,
                  const MatrixXd& inv_M,
                  const VectorXd& p_input,
                  std::function<double(const VectorXd&)> U_func,
                  std::function<VectorXd(const VectorXd&)> dU_func,
                  std::mt19937& rng)
                  {
    
    VectorXd q_proposal = q;
    VectorXd p = p_input;
    VectorXd inv_M_p = inv_M * p;
   // double kinetic_energy = 0.5 * p.dot(inv_M_p);
    double H_curr = U_func(q) + 0.5 * p.dot(inv_M_p);

    for (int l = 0; l < L; ++l) {

        VectorXd grad_U = dU_func(q_proposal);
        p -= 0.5 * epsilon * grad_U;

        q_proposal += epsilon * (inv_M * p);

        grad_U = dU_func(q_proposal);
        p -= 0.5 * epsilon * grad_U;
    }

    VectorXd inv_M_p_proposal = inv_M * p;
    double H_proposal = U_func(q_proposal) + 0.5 * p.dot(inv_M_p_proposal);

    double prob = std::min(1.0, std::exp(H_curr - H_proposal));

    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    if (uniform_dist(rng) < prob) {
        return q_proposal;
    } else {
        return q;
    }
}

// convert MatrixXd to Rcpp::NumericMatrix
Rcpp::NumericMatrix eigen_to_rcpp(const MatrixXd& eigen_mat) {
    Rcpp::NumericMatrix rcpp_mat(eigen_mat.rows(), eigen_mat.cols());
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        for (int j = 0; j < eigen_mat.cols(); ++j) {
            rcpp_mat(i, j) = eigen_mat(i, j);
        }
    }
    return rcpp_mat;
}

// main hmc func exposed t R
// [[Rcpp::export]]
Rcpp::NumericMatrix run_hmc(Rcpp::NumericVector mu_R = R_NilValue,
                           Rcpp::NumericVector sigma_R = R_NilValue,
                           int n_iter = 1000,
                           double epsilon = 0.1,
                           int L = 30,
                           Rcpp::NumericMatrix P_R = R_NilValue,
                           Rcpp::NumericMatrix M_R = R_NilValue,
                           int seed = 123) {

    
    std::mt19937 rng(seed);
    std::normal_distribution<double> p_dist(0.0, 1.0);

    VectorXd mu = Rcpp::as<VectorXd>(mu_R);
    VectorXd sigma = Rcpp::as<VectorXd>(sigma_R);
    MatrixXd P = Rcpp::as<MatrixXd>(P_R);
    MatrixXd M = Rcpp::as<MatrixXd>(M_R);
    MatrixXd inv_M = M.inverse();  
    int dim = mu.size();

    // lambda funcs
    auto U_func = [mu, sigma](const VectorXd& x) { return U(x, mu, sigma); };
    auto dU_func = [mu, sigma](const VectorXd& x) { return dU(x, mu, sigma); };

    VectorXd q(dim);
    for (int i = 0; i < dim; ++i) {
        q(i) = p_dist(rng);
    }

    MatrixXd samples(n_iter, dim);
    
    // do hmc
    for (int iter = 0; iter < n_iter; iter++) {
        VectorXd p = P.row(iter).transpose();
        q = hmc_step(q, epsilon, L, M, inv_M, p, U_func, dU_func,rng);
        for (int col = 0; col < dim; col++) {
            samples(iter, col) = q(col);
        }
    }
    
    // Convert to R matrix and return
    return eigen_to_rcpp(samples);
}
