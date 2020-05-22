import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

np.random.seed(1337)

dpmm_code = """
data{
    int<lower=0> N;   // num samples
    int<lower=0> K;   // num of clusters
    real x[N];        // data
    real alpha;         // concentration
}
parameters {
    real mu[K];   // cluster mean
    real<lower=0,upper=1>    v[K - 1];
    real<lower=0> sigma[K];    // error scale
}
transformed parameters {
    simplex [K] pi;  // mixing weights
  
    // pi[1] = v[1];
    // for(j in 2:(K-1)){
    // 	pi[j] = v[j] * (1 - v[j-1]) * pi[j-1] / v[j-1]; 
    // }
    // pi[K] = 1 - sum(pi[1:(K-1)]);
    
    real summ = 0;
    pi[1] = v[1];
    summ = pi[1];
    for (j in 2:(K - 1)){
        pi[j] = (1.0 - summ) * v[j];
        summ += pi[j];
    }
    pi[K] = 1 - summ;
}
model {
    real a = 1.0;
    real b = 3.0;
    real ps[K];
    sigma ~ inv_gamma(a,b);
    mu ~ normal(0, 5);
    v ~ beta(1, alpha);
    
    for(i in 1:N){
        for(k in 1:K){
            ps[k] = log(pi[k]) + normal_lpdf(x[i] | mu[k], sigma[k]);
        }
        target += log_sum_exp(ps);
    }
}"""

sm = pystan.StanModel(model_code=dpmm_code)

N = 250
x = np.zeros(N)

for n in range(N):
    z = np.argmax(np.random.multinomial(1, np.array([0.1, 0.5, 0.4]), size=1))
    if z == 0:
        x[n] = np.random.normal(loc=-3.0, scale=0.5, size=1)
    elif z == 1:
        x[n] = np.random.normal(loc=0.0, scale=0.75, size=1)
    elif z == 2:
        x[n] = np.random.normal(loc=3.0, scale=1.0, size=1)

dpmm_data = {'N': N, 'K': 10, 'x': x, 'alpha': 1.0}

fit = sm.sampling(data=dpmm_data, iter=1000, chains=1, n_jobs=-1)
print(fit)

fit.plot()
plt.show()
