import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

hmm_code = """
data {
    int<lower=1> K;  // num categories
    int<lower=1> T;  // num steps
    
    real x[T]; // observations
    
    vector<lower=0>[K] alpha;  // prior over init category
    vector<lower=0>[K] beta;  // prior over transition matrix

    real lambda; // prior over emission mean
    real rho;
    
    real nu; // prior over emission coveriance
    real kappa;
}

parameters {
    simplex[K] eta; // prob of init category

    simplex[K] theta[K];  // prob of transition
    
    real mu[K]; // emission mean
    real<lower=0> sigma[K]; // emission covariance
}

model {
    eta ~ dirichlet(alpha);

    for (k in 1:K) 
        theta[k] ~ dirichlet(beta);
    
    for (k in 1:K)
        mu[k] ~ normal(lambda, rho);
    
    for (k in 1:K)
        sigma[k] ~ normal(nu, kappa);
    
    { 
        // forward algorithm computes log p(x|...)
        real tmp[K];
        real gamma[T, K];
        
        for (k in 1:K)
            gamma[1, k] = normal_lpdf(x[1] | mu[k], sigma[k]) + log(eta[k]);
          
        for (t in 2:T) {
            for (k in 1:K) {
                for (j in 1:K)
                    tmp[j] = gamma[t - 1, j] + log(theta[j, k]);
                gamma[t, k] = log_sum_exp(tmp) + normal_lpdf(x[t] | mu[k], sigma[k]);
            }
        }
        target += log_sum_exp(gamma[T]);
    }
}
"""

sm = pystan.StanModel(model_code=hmm_code)

T = 250

z = np.zeros(T, np.int64)
x = np.zeros(T)

z[0] = np.random.binomial(1, 0.9, size=1)

for t in range(T):
    if z[t] == 0:
        x[t] = np.random.normal(loc=2.0, scale=0.1)
        if t < T - 1:
            z[t + 1] = np.random.binomial(1, 0.8, size=1)
    else:
        x[t] = np.random.normal(loc=-2.0, scale=0.1)
        if t < T - 1:
            z[t + 1] = np.random.binomial(1, 0.4, size=1)

hmm_data = {'T': T, 'K': 2, 'x': x,
            'alpha': 2.0 * np.ones(2), 'beta': np.ones(2),
            'lambda': 0.0, 'rho': 1.0,
            'nu': 0.0, 'kappa': 1.0}

fit = sm.sampling(data=hmm_data, iter=5000, chains=2)

print(fit)

fit.plot()
plt.show()
