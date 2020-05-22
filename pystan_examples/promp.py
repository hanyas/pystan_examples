import pystan
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

np.random.seed(1337)
os.environ['STAN_NUM_THREADS'] = "2"  # 2 cores per chain


def generate_basis(T, n):
    h = 0.75
    c = np.linspace(0 - h, 2 * np.pi + h, n)
    psi = np.zeros((T, n))
    idx = 0
    for t in np.nditer(np.linspace(0, 2 * np.pi, T)):
        b = np.exp((-1.0 / (2 * h)) * np.square(t - c))
        b = 1.0 * b / sum(b)
        psi[idx, :] = b
        idx = idx + 1
    return psi


promp_code = """
data {
    int<lower=1> N; // num demos
    int<lower=1> T; // traj length
    int<lower=1> K; // num basis

    vector[T] y[N]; // traj data
    matrix[K, T] psi; // basis feat

    real eps;
}
parameters {
    vector[K] mu;
    
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] L_sigma;

    vector[K] weights[N];
}
model {
    matrix[K, K] L_Sigma;
    L_Sigma = diag_pre_multiply(L_sigma, L_Omega);
    
    mu ~ normal(0, 1);
    L_sigma ~ cauchy(0, 1);
    L_Omega ~ lkj_corr_cholesky(3);
    
    for (n in 1:N) {
        weights[n, :] ~ multi_normal_cholesky(mu, L_Sigma);
        for (t in 1:T) {
            y[n, t] ~ normal(psi[:, t]' * weights[n, :], sqrt(eps));
        }
    }
}
generated quantities {
    matrix[K, K] Omega;
    matrix[K, K] Sigma;
    Omega = multiply_lower_tri_self_transpose(L_Omega);
    Sigma = quad_form_diag(Omega, L_sigma); 
}
"""

extra_compile_args = ['-pthread', '-DSTAN_THREADS']
try:
    sm = pickle.load(open('models/promp.pkl', 'rb'))
except FileNotFoundError:
    sm = pystan.StanModel(model_code=promp_code, extra_compile_args=extra_compile_args)
    with open('models/promp.pkl', 'wb') as f:
        pickle.dump(sm, f)

N, T, K, eps = 15, 100, 6, 5.0

# generate artificial data
K_true = 6
mu_true = np.array([-10.0, 20.0, -12.0, 15.0, -13.0, -5.0])
tmp = np.random.rand(K_true, K_true)
cov_true = tmp @ tmp.T

weights_true = np.random.multivariate_normal(mean=mu_true, cov=cov_true, size=N)

psi = generate_basis(T, K_true).T
y = weights_true @ psi + np.random.multivariate_normal(np.zeros(T), eps * np.eye(T), size=N)

# do regression
psi = generate_basis(T, K).T
weights = y @ np.linalg.pinv(psi.T).T

# plt.subplot(211)
# plt.plot(y.T)
# plt.subplot(212)
# plt.plot(psi.T @ weights.T)
# plt.show()

mu_naive = np.mean(weights, axis=0)
cov_naive = np.cov(weights, rowvar=False)

# do mcmc
promp_data = {'N': N, 'T': T, 'K': K,
              'y': y, 'psi': psi, 'eps': eps}

fit = sm.sampling(data=promp_data, iter=25000, chains=4)
print(fit)

fit.plot()
plt.show()

mu_mcmc = fit.extract(permuted=True)['mu'].mean(axis=0)
cov_mcmc = fit.extract(permuted=True)['Sigma'].mean(axis=0)

print('mu_true', mu_true)
print('mu_naive', mu_naive)
print('mu_mcmc', mu_mcmc)

print('cov_true', np.diag(cov_true))
print('cov_naive', np.diag(cov_naive))
print('cov_mcmc', np.diag(cov_mcmc))
