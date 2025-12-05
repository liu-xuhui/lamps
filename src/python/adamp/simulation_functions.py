import numpy as np

def _component_functions(X):
    return [
        X[:, 0],              
        X[:, 1],                  
        np.sin(np.pi * X[:, 2]),                               
        (X[:, 3]) ** 2,                         
        np.exp(X[:, 4]),                       
        np.log(np.abs(X[:, 5]) + 1),                   
        np.maximum(0, X[:, 6]),       
        (X[:, 7] > 0).astype(float),       
        X[:, 8],                              
        X[:, 9],                            
    ]

def _component_functions_nonadd(X):
    return [
        X[:, 0] * X[:, 1],             
        np.sin(np.pi * X[:, 2]),            
        (X[:, 3]) ** 2,                 
        np.exp(X[:, 4]),                                        
        np.log(np.abs(X[:, 5]) + 1),                 
        np.maximum(0, X[:, 6]),      
        (X[:, 7] > 0).astype(float),    
        X[:, 8],                              
        X[:, 9],                               
    ]

def SimuFriedmanAdditive(N, M, N1, snr=1, seed=110, corr = 0.9, permute = False):
    np.random.seed(seed)

    rng = np.random.default_rng(seed)
    cov = corr ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))

    # this two rows are used to permute cov
    if permute and corr != 0:
        perm = rng.permutation(M)
        cov = cov[np.ix_(perm, perm)]

    X = rng.multivariate_normal(np.zeros(M), cov, size=N)

    E = np.random.normal(0, 1, size=N)

    comps = _component_functions(X)

    variances = np.array([np.var(c) for c in comps])

    betas = rng.uniform(2, 3, size=10) 

    scaled_betas = betas / np.sqrt(variances)

    scaled_terms = [b * c for b, c in zip(scaled_betas, comps)]
    f_true = np.sum(scaled_terms, axis=0)

    noise_var = np.var(f_true) / snr
    Y = f_true + np.sqrt(noise_var) * E

    # Test data
    X1 = rng.multivariate_normal(np.zeros(M), cov, size=N1)
    E1 = np.random.normal(0, 1, size=N1)
    comps = _component_functions(X1)
    variances = np.array([np.var(c) for c in comps])
    betas = rng.uniform(2, 3, size=10) #* rng.choice([-1,1], size = 10)
    scaled_betas = betas / np.sqrt(variances)
    scaled_terms = [b * c for b, c in zip(scaled_betas, comps)]
    f_true1 = np.sum(scaled_terms, axis=0)
    noise_var = np.var(f_true1) / snr
    Y1 = f_true1 + np.sqrt(noise_var) * E1


    return [X, Y, X1, Y1]

def SimuFriedmanNonAdditive(N, M, N1, snr=1, seed=110, corr = 0.9, interaction_snr = 5, permute = False):
    np.random.seed(seed)

    rng = np.random.default_rng(seed)
    cov = corr ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))

    if permute and corr != 0:
        perm = rng.permutation(M)
        cov = cov[np.ix_(perm, perm)]

    X = rng.multivariate_normal(np.zeros(M), cov, size=N)

    E = np.random.normal(0, 1, size=N)

    comps = _component_functions_nonadd(X)

    variances = np.array([np.var(c) for c in comps])

    betas = rng.uniform(2, 3, size=9)  * np.array([interaction_snr, 1, 1, 1, 1, 1, 1, 1, 1])
    # print(betas)

    scaled_betas = betas / np.sqrt(variances)

    scaled_terms = [b * c for b, c in zip(scaled_betas, comps)]
    f_true = np.sum(scaled_terms, axis=0)

    noise_var = np.var(f_true) / snr
    Y = f_true + np.sqrt(noise_var) * E

    # Test data
    X1 = rng.multivariate_normal(np.zeros(M), cov, size=N1)
    E1 = np.random.normal(0, 1, size=N1)
    comps = _component_functions_nonadd(X1)
    variances = np.array([np.var(c) for c in comps])
    betas = rng.uniform(0, 1, size=9) 
    scaled_betas = betas / np.sqrt(variances)
    scaled_terms = [b * c for b, c in zip(scaled_betas, comps)]
    f_true1 = np.sum(scaled_terms, axis=0)
    noise_var = np.var(f_true1) / snr
    Y1 = f_true1 + np.sqrt(noise_var) * E1

    return [X, Y, X1, Y1]


def SimuLinear(N, M, N1, k=10, rho=0.9, snr=1, seed=110):
    """
    Simulate a linear model where stability selection may fail due to strong correlation among features.

    Parameters:
    N: int - Number of training samples
    M: int - Number of features
    N1: int - Number of test samples
    k: int - Number of informative features (located at the start)
    rho: float - Correlation coefficient between features
    seed: int - Random seed for reproducibility

    Returns:
    X: Training features (N x M)
    Y: Training responses (N,)
    X1: Test features (N1 x M)
    Y1: Test responses (N1,)
    """
    b = np.sqrt(snr)

    rng = np.random.default_rng(seed)

    # Generate Toeplitz covariance matrix
    cov = rho ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))

    # Training data
    X = rng.multivariate_normal(np.zeros(M), cov, size=N)

    beta = np.zeros(M)

    beta[:k] = rng.choice([-1, 1], size=k) * (b + rng.uniform(0, 1, size=k))  # First k features are signals

    Y = X @ beta + rng.normal(scale=1, size=N)  # Add noise to simulate low SNR

    # Test data
    X1 = rng.multivariate_normal(np.zeros(M), cov, size=N1)
    Y1 = X1 @ beta + rng.normal(scale=1, size=N1)

    return [X, Y, X1, Y1]
