import numpy as np
from scipy.stats import ks_2samp, entropy
from scipy.special import softmax

def feature_score(x_ref, x):
    x_ref = entropy(softmax(x_ref))
    x = entropy(softmax(x))
    n_features = 1
    x = x.reshape(x.shape[0], -1)
    x_ref = x_ref.reshape(x_ref.shape[0], -1)
    p_val = np.zeros(n_features, dtype=np.float32)
    dist = np.zeros_like(p_val)
    for f in range(n_features):
        dist[f], p_val[f] = ks_2samp(x_ref[:, f], x[:, f], alternative='two-sided', mode='asymp')
    return p_val, dist

def KSDrift(x_ref, x,  threshold: float = .05, return_p_val=True, return_distance=True):
    p_vals, dist = feature_score(x_ref, x)
    threshold = threshold
    drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
    cd = {}
    cd['is_drift'] = drift_pred
    if return_p_val:
        cd['p_val'] = p_vals
        cd['threshold'] = threshold
    if return_distance:
        cd['distance'] = dist
    return cd

