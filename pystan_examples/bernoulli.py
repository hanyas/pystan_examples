import pystan
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

bernoulli_code = """
data {
    int<lower=0> N;
    int<lower=0, upper=1> y[N];
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ uniform(0, 1);
    y ~ bernoulli(theta);
}
"""

bernoulli_data = {'N': 10, 'y': [1, 0, 0, 1, 0, 0, 0, 0, 1, 1]}

sm = pystan.StanModel(model_code=bernoulli_code)
fit = sm.sampling(data=bernoulli_data, iter=500, chains=2)

print(fit)

theta = fit.extract(permuted=True)['theta']

fit.plot()
plt.show()
