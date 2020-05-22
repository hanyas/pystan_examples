import pystan
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

hmm_code = """
data {
    int<lower=1> K;  // num categories
    int<lower=1> T;  // num steps
    int<lower=1> D;  // state dim
    int<lower=1> R;  // num rollouts

    row_vector[D] x[R, T]; // observations

    vector<lower=0>[K] alpha;  // prior over init category
    vector<lower=0>[K] beta;  // prior over transition matrix

    vector[D] mu0; // prior over emission mean
    real lambda;

    int nu; // prior over emission coveriance
    matrix[D, D] L_kappa; // cholesky factor of inv wishart scale	
}

parameters {
    simplex[K] eta; // prob of init category

    simplex[K] theta[K];  // prob of transition

    vector[D] mu[K]; // emission mean
    
    vector<lower=0>[D] c; // decomposition of emission covariance
    vector[D * (D - 1) / 2] z;
}

model {
    eta ~ dirichlet(alpha);

    for(k in 1:K)
        theta[k] ~ dirichlet(beta);
        
    {
        row_vector[K] gamma[R, T];
        real aux[K];

        matrix[D, D] A;
        int count;
        
        count = 1;
        for (j in 1:(D-1)) {
            for (i in (j+1):D) {
                A[i, j] = z[count];
                count += 1;
            }
            for (i in 1:(j - 1)) {
                A[i, j] = 0.0;
            }
            A[j, j] = sqrt(c[j]);
        }
        for (i in 1:(D-1))
            A[i, D] = 0;
        A[D, D] = sqrt(c[D]);
                
        for (i in 1:D)
            c[i] ~ chi_square(nu - i + 1);
            
        z ~ normal(0, 1);
        
        for(k in 1:K)
            mu[k] ~ multi_normal_cholesky(mu0, 1/lambda * L_kappa * A);

        // forward algorithm computes log p(x|...)
        for (r in 1:R) {
            for (k in 1:K)
                gamma[r, 1, k] = multi_normal_cholesky_lpdf(x[r, 1] | mu[k], L_kappa * A) + log(eta[k]);
            
            for (t in 2:T) {
                for (k in 1:K) {
                    for (j in 1:K)
                        aux[j] = gamma[r, t - 1, j] + log(theta[j, k]);
                    gamma[r, t, k] = log_sum_exp(aux) + multi_normal_cholesky_lpdf(x[r, t] | mu[k], L_kappa * A);
                }
            }
            target += log_sum_exp(gamma[r, T]);
        }
    }
}
"""

sm = pystan.StanModel(model_code=hmm_code)

T = 50
R = 10

z = np.zeros((R, T), np.int64)
x = np.zeros((R, T, 2))

for r in range(R):
    z[r, 0] = np.random.binomial(1, 0.6, size=1)

    for t in range(T):
        if z[r, t] == 0:
            x[r, t, :] = np.random.multivariate_normal(np.array([1.0, 2.0]), 0.01 * np.eye(2))
            if t < T - 1:
                z[r, t + 1] = np.random.binomial(1, 0.8, size=1)
        else:
            x[r, t, :] = np.random.multivariate_normal(np.array([-1.0, -2.0]), 0.01 * np.eye(2))
            if t < T - 1:
                z[r, t + 1] = np.random.binomial(1, 0.4, size=1)

hmm_data = {'T': T, 'R': R,
            'K': 2, 'D': 2, 'x': x,
            'alpha': np.ones(2), 'beta': np.ones(2),
            'mu0': np.zeros(2), 'lambda': 1.0,
            'nu': 5,
            'L_kappa': np.linalg.cholesky(np.eye(2))}

fit = sm.sampling(data=hmm_data, iter=5000, chains=1)
print(fit)

fit.plot()
plt.show()
