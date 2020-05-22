import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

logisitc_code = """
data {
    int<lower=1> K;
    int<lower=0> N;
    matrix[N,K] x;
    int<lower=0, upper=1> y[N];
}
parameters {
    vector[K] beta;
}
model {
    beta ~ cauchy(0, 3);
    y ~ bernoulli_logit(x * beta);
}
"""

logistic_data = {'K': 2, 'N': 100,
                 'y': np.random.randint(low=0, high=2, size=100),
                 'x': np.random.rand(100, 2)}

sm = pystan.StanModel(model_code=logisitc_code)
fit = sm.sampling(data=logistic_data, iter=1000, chains=4)

print(fit)

beta = fit.extract(permuted=True)['beta']

fit.plot()
plt.show()
