import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

bayesreg_code = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real beta;
    real alpha;
    real<lower=0> sigma;
}
model {
    beta ~ normal(0, 1);
    alpha ~ normal(0, 1);
    sigma ~ normal(0, 1);
    
    y ~ normal(beta * x + alpha, sigma);
}
    
"""

x = np.random.normal(loc=5, scale=10, size=50)

bayesreg_data = {'N': 50,
                 'x': x,
                 'y': 2.0 * x + 3.0 + np.random.randn(50)}

sm = pystan.StanModel(model_code=bayesreg_code)
fit = sm.sampling(data=bayesreg_data, iter=1000, chains=4)

print(fit)

beta = fit.extract(permuted=True)['beta']
alpha = fit.extract(permuted=True)['alpha']
sigma = fit.extract(permuted=True)['sigma']

np.mean(beta, axis=0)
np.mean(alpha, axis=0)
np.mean(sigma, axis=0)

fit.plot()
plt.show()
