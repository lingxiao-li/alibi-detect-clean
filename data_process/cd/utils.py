import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from ..utils.sampling import reservoir_sampling

logger = logging.getLogger(__name__)


def update_reference(X_ref: np.ndarray,
                     X: np.ndarray,
                     n: int,
                     update_method: Dict[str, int] = None,
                     ) -> np.ndarray:

    if isinstance(update_method, dict):
        update_type = list(update_method.keys())[0]
        size = update_method[update_type]
        if update_type == 'reservoir_sampling':
            return reservoir_sampling(X_ref, X, size, n)
        elif update_type == 'last':
            X_update = np.concatenate([X_ref, X], axis=0)
            return X_update[-size:]
        else:
            raise KeyError('Only `reservoir_sampling` and `last` are valid update options for X_ref.')
    else:
        return X_ref


def encompass_batching(
        model: Callable,
        backend: str,
        batch_size: int,
        device: Optional[str] = None,
        preprocess_batch_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_len: Optional[int] = None,
) -> Callable:

    backend = backend.lower()
    kwargs = {'batch_size': batch_size, 'tokenizer': tokenizer, 'max_len': max_len,
              'preprocess_batch_fn': preprocess_batch_fn}
    if backend == 'tensorflow':
        from ..cd.tensorflow.preprocess import preprocess_drift
    elif backend == 'pytorch':
        from ..cd.pytorch.preprocess import preprocess_drift  # type: ignore[no-redef]
        kwargs['device'] = device
    else:
        raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')

    def model_fn(x: Union[np.ndarray, list]) -> np.ndarray:
        return preprocess_drift(x, model, **kwargs)  # type: ignore[arg-type]

    return model_fn


def get_input_shape(shape: Optional[Tuple], x_ref: Union[np.ndarray, list]) -> Optional[Tuple]:
    """ Optionally infer shape from reference data. """
    if isinstance(shape, tuple):
        return shape
    elif hasattr(x_ref, 'shape'):
        return x_ref.shape[1:]
    else:
        logger.warning('Input shape could not be inferred. '
                       'If alibi_detect.models.tensorflow.embedding.TransformerEmbedding '
                       'is used as preprocessing step, a saved detector cannot be reinitialized.')
        return None
