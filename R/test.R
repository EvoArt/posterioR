# load
library(Rcpp)
library(RcppEigen)
library(mvtnorm)
Rcpp::sourceCpp("src/hmc_mvn.cpp")

# prep
n_iter <- 1000
M <- diag(dim)
mu <- c(1.0,3.0)
sigma <- c(1.0,2.0)
dim <- length(mu)
momenta <- rmvnorm(n_iter, mean = rep(0, dim), sigma = M)

# run
samples <- run_hmc(
  mu = mu,
  sigma = sigma,
  n_iter = n_iter,
  epsilon = 0.1,
  L = 30,
  P_R = P_samples,
  M_R = M,
  seed = 123
)

# check
par(mfrow = c(2, 2))
for (i in 1:dim){
  x <- samples[,i]
  plot(x, type = "l", main = "trace", 
       xlab = "Iteration", ylab = "Value")
  hist(x, main = "posterior", probability = TRUE)
  x_range <- seq(min(x), max(x), length = 100)
  true_density <- dnorm(x_range, mean = mu[i], sd = sigma[i])
  lines(x_range, true_density, col = "red", lwd = 2)
}