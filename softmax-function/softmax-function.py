import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)

    if x.ndim ==1:
        x_max = np.max(x)
    else:
        x_max=np.max(x,axis=1, keepdims=True)

    exp_x=np.exp(x-x_max)
    if x.ndim==1:
        sum_exp=np.sum(exp_x)
    else:
        sum_exp=np.sum(exp_x, axis=1, keepdims=True)

    return exp_x/sum_exp