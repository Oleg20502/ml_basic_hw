import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    x0 = np.random.rand(data.shape[0], 1)
    mu = 0
    for i in range(num_steps):
        x1 = data @ x0
        x1 /= np.linalg.norm(x1)
        x0 = x1
    mu = float(np.abs(x1.T @ data @ x1)[0])
    return mu, x1.reshape(data.shape[0])