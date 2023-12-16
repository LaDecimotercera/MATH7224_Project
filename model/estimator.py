import torch
import torch.nn as nn
import numpy as np

class IT_calculator():
    def __init__(self, h=50):
        self.alpha = 1.01
        self.h = h
        # Define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def kernel(self, x, y, sigma=0.1):
        dist_matrix = torch.cdist(x, y) ** 2
        return torch.exp(-dist_matrix / (2 * sigma ** 2))

    def Corr(self, X):
        # Check if X is a tensor
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input X must be a PyTorch tensor.")
        
        # Move X to the defined device (either GPU or CPU)
        X = X.to(self.device)
        n = X.shape[0]
        d = X.shape[1] * X.shape[2]  # Flatten 28x28 to a single dimension
        X_flat = X.view(n, -1)  # Reshape to [n, 28*28]

        sigma = self.h * n ** (-1/(4+d))

        # Use the flattened X for kernel computation
        A = self.kernel(X_flat, X_flat, sigma)
        D = torch.sqrt(A.sum(1)).view(n, 1)
        A = A / (D * D.T)
        
        return A

    def Entropy(self, X):
        eigvals = torch.linalg.eigvalsh(self.Corr(X))
        eigvals = torch.pow(torch.abs(eigvals), self.alpha)
        res = 1/(1 - self.alpha) * torch.log(eigvals.sum()) / np.log(2)
        return res
    
    def JointE(self, X, Y):
        corr_X = self.Corr(X)
        corr_Y = self.Corr(Y)
        joint_matrix = torch.mul(corr_X, corr_Y)
        tr = torch.trace(joint_matrix).item()
        joint_matrix = joint_matrix / tr
        eigvals = torch.linalg.eigvalsh(joint_matrix)
        eigvals = torch.pow(torch.abs(eigvals), self.alpha)
        res = 1/(1 - self.alpha) * torch.log(eigvals.sum()) / np.log(2)
        return res

    def MI(self, X, Y):
        res = self.Entropy(X) + self.Entropy(Y) - self.JointE(X, Y)
        return res
