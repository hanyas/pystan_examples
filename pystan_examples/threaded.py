import pystan
import os
import time

# set environmental variable STAN_NUM_THREADS
# Use 4 cores per chain
os.environ['STAN_NUM_THREADS'] = "4"

# Example model
# see http://discourse.mc-stan.org/t/cant-make-cmdstan-2-18-in-windows/5088/18
stan_code = """
functions {
  vector bl_glm(vector mu_sigma, vector beta,
                real[] x, int[] y) {
    vector[2] mu = mu_sigma[1:2];
    vector[2] sigma = mu_sigma[3:4];
    real lp = normal_lpdf(beta | mu, sigma);
    real ll = bernoulli_logit_lpmf(y | beta[1] + beta[2] * to_vector(x));
    return [lp + ll]';
  }
}
data {
  int<lower = 0> K;
  int<lower = 0> N;
  vector[N] x;
  int<lower = 0, upper = 1> y[N];
}
transformed data {
  int<lower = 0> J = N / K;
  real x_r[K, J];
  int<lower = 0, upper = 1> x_i[K, J];
  {
    int pos = 1;
    for (k in 1:K) {
      int end = pos + J - 1;
      x_r[k] = to_array_1d(x[pos:end]);
      x_i[k] = y[pos:end];
      pos += J;
    }
  }
}
parameters {
  vector[2] beta[K];
  vector[2] mu;
  vector<lower=0>[2] sigma;
}
model {
  mu ~ normal(0, 2);
  sigma ~ normal(0, 2);
  target += sum(map_rect(bl_glm, append_row(mu, sigma),
                         beta, x_r, x_i));
}
"""
stan_data = dict(K=4, N=12,
                 x=[1.204, -0.573, -1.35, -1.157,
                    -1.29, 0.515, 1.496, 0.918,
                    0.517, 1.092, -0.485, -2.157],
                 y=[1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

extra_compile_args = ['-pthread', '-DSTAN_THREADS']
stan_model = pystan.StanModel(
    model_code=stan_code,
    extra_compile_args=extra_compile_args
)

# use the default 4 chains == 4 parallel process
# used cores = min(cpu_cores, 4*STAN_NUM_THREADS)
t1 = time.time()
fit = stan_model.sampling(data=stan_data, n_jobs=4, chains=10)
print(time.time() - t1)

print(fit)
