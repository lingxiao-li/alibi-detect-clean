from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import tensorflow as tf
from ...utils.tensorflow.prediction import predict_batch


def preprocess_drift(x: Union[np.ndarray, list], model: tf.keras.Model,
                     preprocess_batch_fn: Callable = None, tokenizer: Callable = None,
                     max_len: int = None, batch_size: int = int(1e10), dtype: Type[np.generic] = np.float32) \
        -> Union[np.ndarray, tf.Tensor]:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x
        Batch of instances.
    model
        Model used for preprocessing.
    preprocess_batch_fn
        Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
        processed by the TensorFlow model.
    tokenizer
        Optional tokenizer for text drift.
    max_len
        Optional max token length for text drift.
    batch_size
        Batch size.
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array with predictions.
    """
    return predict_batch(x, model, batch_size=batch_size, preprocess_fn=preprocess_batch_fn, dtype=dtype)
