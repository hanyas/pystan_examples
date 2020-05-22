import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

gmm_code = """
data {
    int<lower=0> N;
    int<lower=1> K;
    
    vector[N] x;
    
    real mu_prior_mean;
    real mu_prior_cov;
    
    real sigma_prior_mean;
    real sigma_prior_cov;
}

parameters {
    real mu[K];
    real<lower=0> sigma[K];
    
    real<lower=0, upper=1> alpha;
}

model {
    alpha ~ uniform(0, 1);
    
    mu[1] ~ normal(mu_prior_mean, mu_prior_cov);
    mu[2] ~ normal(mu_prior_mean, mu_prior_cov);
    
    sigma[1] ~ normal(sigma_prior_mean, sigma_prior_cov);
    sigma[2] ~ normal(sigma_prior_mean, sigma_prior_cov);

    for (n in 1:N) {
        real lp1; real lp2;
        
        lp1 = bernoulli_lpmf(0| alpha) + normal_lpdf(x[n]| mu[1], sigma[1]);
        lp2 = bernoulli_lpmf(1| alpha) + normal_lpdf(x[n]| mu[2], sigma[2]);
        
        target += log_sum_exp(lp1, lp2);
    }
}
"""

sm = pystan.StanModel(model_code=gmm_code)

N = 100
x = np.zeros(N)

for n in range(N):
    z = np.random.binomial(1, 0.7, size=1)
    if z == 0:
        x[n] = np.random.normal(loc=2.0, scale=0.05, size=1)
    else:
        x[n] = np.random.normal(loc=-2.0, scale=0.2, size=1)

gmm_data = {'N': N, 'K': 2, 'x': x,
            'mu_prior_mean': 0.0, 'mu_prior_cov': 1.0,
            'sigma_prior_mean': 1.0, 'sigma_prior_cov': 1.0}

fit = sm.sampling(data=gmm_data, iter=1000, chains=1)

print(fit)

mu = fit.extract(permuted=True)['mu']
sigma = fit.extract(permuted=True)['sigma']
alpha = fit.extract(permuted=True)['alpha']

fit.plot()
plt.show()
