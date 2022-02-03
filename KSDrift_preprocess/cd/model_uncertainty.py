import logging
import numpy as np
from typing import Callable, Dict, Optional, Union
from functools import partial
from .ks import KSDrift
from .preprocess import classifier_uncertainty
from .utils import encompass_batching
from ..utils.frameworks import has_pytorch, has_tensorflow

logger = logging.getLogger(__name__)


class DataPreprocess:
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            model: Callable,
            p_val: float = .05,
            backend: Optional[str] = None,
            update_x_ref: Optional[Dict[str, int]] = None,
            preds_type: str = 'probs',
            uncertainty_type: str = 'entropy',
            margin_width: float = 0.1,
            batch_size: int = 32,
            preprocess_batch_fn: Optional[Callable] = None,
            device: Optional[str] = None,
            tokenizer: Optional[Callable] = None,
            max_len: Optional[int] = None,
            data_type: Optional[str] = None,
    ) -> None:

        if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
            raise ImportError(f'{backend} not installed. Cannot initialize and run the '
                              f'ClassifierUncertaintyDrift detector with {backend} backend.')

        if backend is None:
            if device not in [None, 'cpu']:
                raise NotImplementedError('Non-pytorch/tensorflow models must run on cpu')
            model_fn = model
        else:
            model_fn = encompass_batching(
                model=model,
                backend=backend,
                batch_size=batch_size,
                device=device,
                preprocess_batch_fn=preprocess_batch_fn,
                tokenizer=tokenizer,
                max_len=max_len
            )

        preprocess_fn = partial(
            classifier_uncertainty,
            model_fn=model_fn,
            preds_type=preds_type,
            uncertainty_type=uncertainty_type,
            margin_width=margin_width,
        )

        self._detector: Union[KSDrift]

        if uncertainty_type == 'entropy':
            self._detector = KSDrift(
                x_ref=x_ref,
                p_val=p_val,
                preprocess_x_ref=True,
                update_x_ref=update_x_ref,
                preprocess_fn=preprocess_fn,
                data_type=data_type
            )
        else:
            raise NotImplementedError("Only uncertainty types 'entropy' or 'margin' supported.")

        self.meta = self._detector.meta
        self.meta['name'] = 'ClassifierUncertaintyDrift'

    def preprocess(self, x: Union[np.ndarray, list]):
        return self._detector.preprocess(x)
