import numpy as np

# Step 1: Initialize Variables using SVD
def initialize_variables(users, items, latent_factors, R):
    U_svd, sigma_svd, Vt_svd = np.linalg.svd(R, full_matrices=False)
    U = U_svd[:, :latent_factors]
    V = np.dot(np.diag(sigma_svd[:latent_factors]), Vt_svd[:latent_factors, :]).T
    return U, V

# Step 2: Calculate Error
def calculate_error(R, U, V):
    predicted_ratings = np.dot(U, V.T)
    error = np.mean((R - predicted_ratings)**2)
    return error

# Step 3: Compute Gradients
def compute_gradients(R, U, V):
    grad_U = -2 * np.dot((R - np.dot(U, V.T)), V)
    grad_V = -2 * np.dot((R - np.dot(U, V.T)).T, U)
    return grad_U, grad_V

# Step 4: Update Parameters
def update_parameters(U, V, grad_U, grad_V, learning_rate):
    U -= learning_rate * grad_U
    V -= learning_rate * grad_V
    return U, V

# Step 5: Repeat
def matrix_factorization(R, latent_factors, learning_rate, iterations):
    users, items = R.shape
    U, V = initialize_variables(users, items, latent_factors, R)
    
    for i in range(iterations):
        error = calculate_error(R, U, V)
        if i%10 == 0:
            print(f"Iteration {i+1}, Error: {error}")
        
        grad_U, grad_V = compute_gradients(R, U, V)
        U, V = update_parameters(U, V, grad_U, grad_V, learning_rate)
    
    return U, V

import numpy as np

# Example user-item matrix (ratings matrix)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

U, V = matrix_factorization(R, latent_factors=10, learning_rate=0.01, iterations=100)

print("\nPredicted_Matrix\n")

print(np.dot(U,V.T))
