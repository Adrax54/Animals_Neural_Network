import numpy as np

# Funkcja błędu i jej pochodna
def mse(y_true, y_pred):
    return  np.mean(np.power(y_true-y_pred, 2))

def mse_prim(y_true, y_pred):
    return -((y_true - y_pred) * y_pred* (1-y_pred))

