import logging
import numpy as np
from typing import Callable, Dict, Optional, Union
from functools import partial
from alibi_detect.cd.ks import KSDrift
from alibi_detect.cd.preprocess import classifier_uncertainty
from alibi_detect.cd.utils import encompass_batching
from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow

logger = logging.getLogger(__name__)


class ClassifierUncertaintyDrift:
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
        """
        Test for a change in the number of instances falling into regions on which the model is uncertain.
        Performs either a K-S test on prediction entropies or Chi-squared test on 0-1 indicators of predictions
        falling into a margin of uncertainty (e.g. probs falling into [0.45, 0.55] in binary case).

        Parameters
        ----------
        x_ref
            Data used as reference distribution. Should be disjoint from the data the model was trained on
            for accurate p-values.
        model
            Classification model outputting class probabilities (or logits)
        backend
            Backend to use if model requires batch prediction. Options are 'tensorflow' or 'pytorch'.
        p_val
            p-value used for the significance of the test.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preds_type
            Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]).
        uncertainty_type
            Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'.
        margin_width
            Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance
            if the highest two class probabilities it assigns to the instance differ by less than margin_width.
        batch_size
            Batch size used to evaluate model. Only relevant when backend has been specified for batch prediction.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the model.
        device
            Device type used. The default None tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.
        tokenizer
            Optional tokenizer for NLP models.
        max_len
            Optional max token length for NLP models.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """

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

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True) -> Dict[Dict[str, str], Dict[str, Union[np.ndarray, int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return the corresponding test statistic (K-S for 'entropy', Chi2 for 'margin').

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the p-value, threshold and test statistic.
        """
        return self._detector.predict(x, return_p_val=return_p_val, return_distance=return_distance)