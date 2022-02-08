import numpy as np
from scipy.stats import chi2_contingency

def feature_score(x_ref, x, x_ref_categories):
    n_features = 1
    x_ref = x_ref.reshape(x_ref.shape[0], -1)
    x = x.reshape(x.shape[0], -1)

    # apply counts on union of categories per variable in both the reference and test data
    x_categories = {f: list(np.unique(x[:, f])) for f in range(n_features)}
    all_categories = {f: list(set().union(x_ref_categories[f], x_categories[f]))  # type: ignore
                        for f in range(n_features)}
    x_ref_count = get_counts(x_ref, all_categories)
    x_count = get_counts(x, all_categories)

    p_val = np.zeros(n_features, dtype=np.float32)
    dist = np.zeros_like(p_val)
    for f in range(n_features):  # apply Chi-Squared test
        contingency_table = np.vstack((x_ref_count[f], x_count[f]))
        dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
    return p_val, dist

def get_counts(x, categories):
    return {f: [(x[:, f] == v).sum() for v in vals] for f, vals in categories.items()}

def ChiSquareDrift(x_ref, x, x_ref_categories, threshold: float = .05, return_p_val=True, return_distance=True):
    p_vals, dist = feature_score(x_ref, x, x_ref_categories)
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