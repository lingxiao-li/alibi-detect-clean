import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from typing import Callable, Union


def classifier_uncertainty(
    x: Union[np.ndarray, list],
    model_fn: Callable,
    preds_type: str = 'probs',
    uncertainty_type: str = 'entropy',
    margin_width: float = 0.1,
) -> np.ndarray:

    preds = model_fn(x)

    if preds_type == 'probs':
        if np.abs(1 - np.sum(preds, axis=-1)).mean() > 1e-6:
            raise ValueError("Probabilities across labels should sum to 1")
        probs = preds
    elif preds_type == 'logits':
        probs = softmax(preds, axis=-1)
    else:
        raise NotImplementedError("Only prediction types 'probs' and 'logits' supported.")

    if uncertainty_type == 'entropy':
        uncertainties = entropy(probs, axis=-1)
    elif uncertainty_type == 'margin':
        top_2_probs = -np.partition(-probs, kth=1, axis=-1)[:, :2]
        diff = top_2_probs[:, 0] - top_2_probs[:, 1]
        uncertainties = (diff < margin_width).astype(int)
    else:
        raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported")

    return uncertainties[:, None]  # Detectors expect N x d

