import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x= np.array(x)
    # Write code here
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    return (exp_x - exp_neg_x)/(exp_x + exp_neg_x)
