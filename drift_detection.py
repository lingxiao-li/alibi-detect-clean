#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:27:34 2022

@author: lingxiaoli
"""

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, entropy
from scipy.special import softmax


class ChiSquareDrift:

    def __init__(self, x_ref, threshold: float = .05, return_p_val=True, return_distance=True):
        self.x_ref = x_ref
        self.threshold = threshold
        self.return_p_val = return_p_val
        self.return_distance = return_distance

    def updata_ref(self, x_ref):
        self.x_ref = x_ref
    
    def update_threshold(self, threshold):
        self.threshold = threshold

    def update_return_p_val(self, return_p_val):
        self.return_p_val = return_p_val

    def update_return_distance(self, return_distance):
        self.return_distance = return_distance

    def process_data(self, x):
        margin_width = 0.1
        temp = softmax(x.detach().numpy(), axis=-1)
        top_2_probs = -np.partition(-temp, kth=1, axis=-1)[:, :2]
        diff = top_2_probs[:, 0] - top_2_probs[:, 1]
        x_logist = (diff < margin_width).astype(int)
        return x_logist[:, None]

    def feature_score_Chi(self, x):
        x_ref = self.process_data(self.x_ref)
        x = self.process_data(x)
        x_ref_categories = {0: [0, 1]}
        n_features = 1
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        # apply counts on union of categories per variable in both the reference and test data
        x_categories = {f: list(np.unique(x[:, f])) for f in range(n_features)}
        all_categories = {f: list(set().union(x_ref_categories[f], x_categories[f]))  # type: ignore
                            for f in range(n_features)}
        x_ref_count = self.get_counts(x, all_categories)
        x_count = self.get_counts(x, all_categories)

        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):  # apply Chi-Squared test
            contingency_table = np.vstack((x_ref_count[f], x_count[f]))
            dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
        return p_val, dist

    def get_counts(self, x, categories):
        return {f: [(x[:, f] == v).sum() for v in vals] for f, vals in categories.items()}

    def get_result(self, x):
        p_vals, dist = self.feature_score_Chi(x)
        threshold = self.threshold
        drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        cd = {}
        cd['is_drift'] = drift_pred
        if self.return_p_val:
            cd['p_val'] = p_vals
            cd['threshold'] = threshold
        if self.return_distance:
            cd['distance'] = dist
        return cd

class KSDrift:

    def __init__(self, x_ref, threshold: float = .05, return_p_val=True, return_distance=True):
        self.x_ref = x_ref
        self.threshold = threshold
        self.return_p_val = return_p_val
        self.return_distance = return_distance

    def updata_ref(self, x_ref):
        self.x_ref = x_ref

    def updata_ref(self, x_ref):
        self.x_ref = x_ref
    
    def update_threshold(self, threshold):
        self.threshold = threshold

    def update_return_p_val(self, return_p_val):
        self.return_p_val = return_p_val

    def update_return_distance(self, return_distance):
        self.return_distance = return_distance


    def feature_score_KS(self, x):
        x_ref = entropy(softmax(self.x_ref.detach().numpy(), axis=-1), axis=-1)
        x = entropy(softmax(x.detach().numpy(), axis=-1), axis=-1)
        n_features = 1
        x = x.reshape(x.shape[0], -1)
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):
            dist[f], p_val[f] = ks_2samp(x_ref[:, f], x[:, f], alternative='two-sided', mode='asymp')
        return p_val, dist

    def get_result(self, x):
        p_vals, dist = self.feature_score_KS(x)
        threshold = self.threshold
        drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        cd = {}
        cd['is_drift'] = drift_pred
        if self.return_p_val:
            cd['p_val'] = p_vals
            cd['threshold'] = threshold
        if self.return_distance:
            cd['distance'] = dist
        return cd

def drift_detection(x_ref, threshold: float = .05, 
                    return_p_val=True, return_distance=True, method='KSDrift'):
    if method == 'KSDrift':
        return KSDrift(x_ref, threshold, return_p_val, return_distance)
    elif method == "ChiSquareDrift":
        return ChiSquareDrift(x_ref, threshold, return_p_val, return_distance)