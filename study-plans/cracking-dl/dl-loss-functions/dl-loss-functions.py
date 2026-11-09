import numpy as np

def loss_functions(y_true: list, y_pred: list, loss_type: str) -> float:
    targets=np.asarray(y_true)
    predictions=np.asarray(y_pred,dtype=np.float64)
    if loss_type=="mse": value=np.mean((targets.astype(np.float64)-predictions)**2)
    elif loss_type=="bce":
        probabilities=np.clip(predictions,1e-15,1.0-1e-15); value=-np.mean(targets*np.log(probabilities)+(1-targets)*np.log(1.0-probabilities))
    elif loss_type=="hinge": value=np.mean(np.maximum(0.0,1.0-targets*predictions))
    else:
        shifted=predictions-np.max(predictions,axis=1,keepdims=True); log_sum_exp=np.log(np.sum(np.exp(shifted),axis=1)); value=np.mean(log_sum_exp-shifted[np.arange(len(targets)),targets.astype(int)])
    return round(float(value),4)