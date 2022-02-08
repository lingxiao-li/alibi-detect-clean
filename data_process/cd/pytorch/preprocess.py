from typing import Callable, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from ...utils.pytorch.prediction import predict_batch


def preprocess_drift(x: Union[np.ndarray, list], model: Union[nn.Module, nn.Sequential],
                     device: Optional[torch.device] = None, preprocess_batch_fn: Callable = None,
                     tokenizer: Optional[Callable] = None, max_len: Optional[int] = None,
                     batch_size: int = int(1e10), dtype: Union[Type[np.generic], torch.dtype] = np.float32) \
        -> Union[np.ndarray, torch.Tensor, tuple]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    preprocess_batch_fn
        Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
        processed by the PyTorch model.
    tokenizer
        Optional tokenizer for text drift.
    max_len
        Optional max token length for text drift.
    batch_size
        Batch size used during prediction.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array or torch tensor with predictions.
    """
    return predict_batch(x, model, device=device, batch_size=batch_size,
                         preprocess_fn=preprocess_batch_fn, dtype=dtype)