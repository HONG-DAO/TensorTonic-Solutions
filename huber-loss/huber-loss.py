import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    e = y_true - y_pred
    norm_e = np.abs(e)

    # Fixed the 'int' object is not callable by adding * after delta
    loss = np.where(norm_e <= delta, 
                    0.5 * (e**2), 
                    delta * (norm_e - 0.5 * delta))
    return np.mean(loss)