import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for i in range(steps):
        # Du doan xac suat p_hat (forward pass)
        z = np.dot(X, w) + b
        p_hat = _sigmoid(z)

        # Tinh loss function
        loss = p_hat - y

        # gradient 
        dw = (1/N)*np.dot(X.T, loss)
        db = np.mean(loss)

        # update weight
        w = w - lr*dw
        b = b - lr*db

    return w,b