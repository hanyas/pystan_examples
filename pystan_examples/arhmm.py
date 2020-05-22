import matplotlib.pyplot as plt
import numpy as np
import pystan

np.set_printoptions(precision=5, suppress=True)

plt.style.use('seaborn-darkgrid')

arhmm_code = """
data {
    int<lower=1> T;  // num steps
    int<lower=1> K;  // num categories
    int<lower=1> D;  // dim of obs
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

    vector[D] imu; // init emission mean

    matrix[D, D] mat[K]; // emit prob matrix
    
    vector<lower=0>[D] c; // decomposition of emission covariance
    vector[D * (D - 1) / 2] z;
}

model {
    eta ~ dirichlet(alpha);

    for (k in 1:K)
        theta[k] ~ dirichlet(beta);

    for(k in 1:K)
        for (d in 1:D)
            mat[k, d,] ~ normal(0, 1);

    {
        row_vector[K] gamma[R, T];
        real aux[K];
        vector[D] mu;

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
        
        imu ~ multi_normal_cholesky(mu0, 1/lambda * L_kappa * A);

        // forward algorithm computes log p(x|...)
        for (r in 1:R) {
            for (k in 1:K)
                gamma[r, 1, k] = multi_normal_cholesky_lpdf(x[r, 1] | imu, L_kappa * A) + log(eta[k]);
              
            for (t in 2:T) {
                for (k in 1:K) {
                    for (j in 1:K)
                        aux[j] = gamma[r, t - 1, j] + log(theta[j, k]);
                        
                    mu = mat[k] * x[r, t - 1]';
                    gamma[r, t, k] = log_sum_exp(aux) + multi_normal_cholesky_lpdf(x[r, t] | mu', L_kappa * A);
                }
            }
            target += log_sum_exp(gamma[r, T]);
        }
    }
}
"""

sm = pystan.StanModel(model_code=arhmm_code, model_name="arhmm")

T = 50
R = 10

z = np.zeros((R, T), np.int64)
x = np.zeros((R, T, 2))

A = np.zeros((2, 2, 2))
A[:, :, 0] = np.array([[1.0, 0.01], [-0.25, 0.995]])
A[:, :, 1] = np.array([[1.0, 0.01], [-1.0, 0.99]])

for r in range(R):
    z[r, 0] = np.random.binomial(1, 0.8, size=1)
    x[r, 0, :] = np.random.multivariate_normal(np.array([1.0, 0.0]), 0.01 * np.eye(2))

    for t in range(1, T):
        if z[r, t - 1] == 0:
            z[r, t] = np.random.binomial(1, 0.3, size=1)
        else:
            z[r, t] = np.random.binomial(1, 0.9, size=1)

        if z[r, t] == 0:
            x[r, t, :] = np.random.multivariate_normal(np.dot(A[:, :, 0], x[r, t - 1, :]), 0.01 * np.eye(2))
        else:
            x[r, t, :] = np.random.multivariate_normal(np.dot(A[:, :, 1], x[r, t - 1, :]), 0.01 * np.eye(2))

arhmm_data = {'T': T, 'R': R, 'K': 2, 'D': 2, 'x': x,
              'alpha': np.ones(2), 'beta': np.ones(2),
              'mu0': np.zeros(2), 'lambda': 1.0,
              'nu': 5, 'L_kappa': np.linalg.cholesky(np.eye(2))}

# fit = sm.sampling(data=arhmm_data, warmup=5000, iter=10000, chains=1)
# print(fit)

# fit.plot()
# plt.show()

opt = sm.optimizing(data=arhmm_data)
print(opt)
