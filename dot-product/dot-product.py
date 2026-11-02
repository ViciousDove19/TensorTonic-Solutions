import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    ans = 0
    if len(x) == len(y):
        for i in range(len(x)):
            ans += x[i]*y[i]
            
    return ans*1.0