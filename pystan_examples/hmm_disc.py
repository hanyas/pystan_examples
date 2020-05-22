import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

hmm_code_naive = """
data {
    int<lower=0> T; // num time steps
    int<lower=1> K; // num categories
    int<lower=1> N; // num emission states
    
    int<lower=1, upper=N> x[T]; // emissions
    int<lower=1, upper=K> z[T]; // categories
    
    vector<lower=0>[K] alpha; // trans prior
    vector<lower=0>[N] beta; // emit prior
    vector<lower=0>[K] zeta; // init prior
}
parameters {
    simplex[K] theta[K]; // trans prob
    simplex[N] phi[K]; // emit prob
    simplex[K] eta; // init prob
}
model {
    eta ~ dirichlet(zeta);
    
    for(k in 1:K)
        theta[k] ~ dirichlet(alpha);
        
    for(k in 1:K)
        phi[k] ~ dirichlet(beta);
    
    z[1] ~ categorical(eta);
    
    for(t in 1:T)
        x[t] ~ categorical(phi[z[t]]);
    
    for(t in 2:T)
        z[t] ~ categorical(theta[z[t - 1]]);
}
"""

sm = pystan.StanModel(model_code=hmm_code_naive)

T = 100
z = np.zeros(T, np.int64)
z[0] = np.random.binomial(1, 0.7, size=1) + 1

x = np.zeros(T, np.int64)

for t in range(T):
    if z[t] == 1:
        x[t] = np.random.binomial(1, 0.9) + 1
        if t < T - 1:
            z[t + 1] = np.random.binomial(1, 0.8, size=1) + 1
    else:
        x[t] = np.random.binomial(1, 0.4) + 1
        if t < T - 1:
            z[t + 1] = np.random.binomial(1, 0.2, size=1) + 1

hmm_data = {'T': T, 'K': 2, 'N': 2,
            'x': x, 'z': z,
            'alpha': np.ones(2),
            'beta': np.ones(2),
            'zeta': np.ones(2)}

fit = sm.sampling(data=hmm_data, iter=10000, chains=4)

print(fit)

fit.plot()
plt.show()
