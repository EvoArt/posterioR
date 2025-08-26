#include <Rcpp.h>
#include <cmath> 
using namespace Rcpp;

// HMC bits converted from Julia
// negative log density
double U(double x, double mu = 1.0, double sigma = 2.0) {
    return -std::log(sigma) + 0.5 * std::log(2 * M_PI) 
           + 0.5 * std::pow((x - mu) / sigma, 2);
}
// Gradient thereof
double dU(double x, double mu = 1.0, double sigma = 2.0) {
    return (x - mu) / (sigma * sigma);
}

//' Simple HMC Sampler
 //' 
 //' @param n_samples Number of MCMC samples to generate
 //' @param target_log_density R function that computes log density of target distribution
 //' @param initial_value Initial value for the chain
 //' @param proposal_sd Standard deviation for proposal distribution
 //' 
 //' @return List containing samples and acceptance rate
 //' @export
 // [[Rcpp::export]]
 List hmc(int n_samples, 
                  Function target_log_density,
                  double initial_value,
                  double proposal_sd) {
                    
   
   // Storage for samples
   NumericVector samples(n_samples);
   
   // Initialize chain
   double current_x = initial_value;
   double current_log_dens = as<double>(target_log_density(current_x));
   
   // Counter for accepted proposals
   int n_accepted = 0;
   
   // Main MCMC loop
   for(int i = 0; i < n_samples; i++) {
     
     // Generate proposal: current + normal(0, proposal_sd)
     double proposal = current_x + R::rnorm(0, proposal_sd);
     
     // Evaluate log density at proposal
     double proposal_log_dens = as<double>(target_log_density(proposal));
     
     // Calculate acceptance probability (in log scale)
     double log_alpha = proposal_log_dens - current_log_dens;
     
     // Accept or reject
     if(log(R::runif(0, 1)) < log_alpha) {
       // Accept the proposal
       current_x = proposal;
       current_log_dens = proposal_log_dens;
       n_accepted++;
     }
     // If rejected, current_x stays the same
     
     // Store current state
     samples[i] = current_x;
   }
   
   // Calculate acceptance rate
   double acceptance_rate = (double)n_accepted / n_samples;
   
   // Return results
   return List::create(
     Named("samples") = samples,
     Named("acceptance_rate") = acceptance_rate
   );
 }