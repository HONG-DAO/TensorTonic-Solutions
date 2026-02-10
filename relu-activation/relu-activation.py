import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.atleast_1d(x)
    # Write code here

    return np.array(np.maximum(0,x))