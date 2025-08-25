library(Rcpp)

# Compile the C++ 
sourceCpp("src/simple_mcmc.cpp")


# targ Normal(mean=2, sd=1.5)
normal_log_density <- function(x) {
  dnorm(x, mean = 2, sd = 1.5, log = TRUE)
}

result <- simple_mcmc(
  n_samples = 1000,
  target_log_density = normal_log_density,
  initial_value = 0,
  proposal_sd = 1.0
)


# res
par(mfrow = c(1, 2))
plot(result$samples, type = "l", main = "trace", 
     xlab = "Iteration", ylab = "Value")
hist(result$samples, main = "posterior", probability = TRUE)

# Overlay true density
x_range <- seq(min(result$samples), max(result$samples), length = 100)
true_density <- dnorm(x_range, mean = 2, sd = 1.5)
lines(x_range, true_density, col = "red", lwd = 2)
