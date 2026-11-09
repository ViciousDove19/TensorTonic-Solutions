import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X = np.asarray(X)
    y = [int(i) for i in y]

    W = np.zeros(X.shape[1], dtype = np.float64)
    b = 0.0

    for _ in range(epochs):
        for x,yi in zip(X,y):
            z = W@x + b
            if z >= 0:
                pred = 1
            else:
                pred = 0
            W = W + lr*x*(yi-pred)
            b = b + lr*(yi-pred)
        
    ans = []
    ans.append(W)
    ans.append(b)

    return ans
