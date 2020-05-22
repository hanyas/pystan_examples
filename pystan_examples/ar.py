import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

ar_code = """
data {
    int<lower=0> N;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real sigma;
}
model {
    beta ~ normal(0, 10);
    alpha ~ normal(0, 10);
    sigma ~ normal(0, 10);

    for(n in 2:N)
        y[n] ~ normal(alpha + beta * y[n - 1], sigma);
}
"""

ar_data = {'N': 100, 'y': np.random.rand(100)}

sm = pystan.StanModel(model_code=ar_code)
fit = sm.sampling(data=ar_data, iter=1000, chains=4)

print(fit)

beta = fit.extract(permuted=True)['beta']
alpha = fit.extract(permuted=True)['alpha']
sigma = fit.extract(permuted=True)['sigma']

fit.plot()
plt.show()
