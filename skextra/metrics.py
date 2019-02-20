import numpy as np

def rmse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def mcr(y_true,y_pred):
    return 1-np.mean(y_true==y_pred)
