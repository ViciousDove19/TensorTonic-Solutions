import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    values=np.asarray(X,dtype=np.float64)
    if mode=="train": values=values*np.asarray(mask,dtype=np.float64)/(1.0-drop_prob)
    return [[round(float(v),4) for v in row] for row in values]