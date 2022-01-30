import logging
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ..base import BaseDetector, concept_drift_dict
from .utils import get_input_shape, update_reference
from ..utils.frameworks import has_pytorch, has_tensorflow
from ..utils.statstest import fdr


if has_pytorch:
    import torch

if has_tensorflow:
    import tensorflow as tf

logger = logging.getLogger(__name__)

class BaseUnivariateDrift(BaseDetector):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Generic drift detector component which serves as a base class for methods using
        univariate tests. If n_features > 1, a multivariate correction is applied such that
        the false positive rate is upper bounded by the specified p-value, with equality in
        the case of independent features.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for significance of the statistical test for each feature. If the FDR correction method
            is used, this corresponds to the acceptable q-value.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
            Typically a dimensionality reduction technique.
        correction
            Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False Discovery Rate).
        n_features
            Number of features used in the statistical test. No need to pass it if no preprocessing takes place.
            In case of a preprocessing step, this can also be inferred automatically but could be more
            expensive to compute.
        input_shape
            Shape of input data. Needs to be provided for text data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # optionally already preprocess reference data
        self.p_val = p_val
        if preprocess_x_ref and isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref
        self.preprocess_x_ref = preprocess_x_ref
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.correction = correction
        self.n = len(x_ref)

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # compute number of features for the univariate tests
        if isinstance(n_features, int):
            self.n_features = n_features
        elif not isinstance(preprocess_fn, Callable) or preprocess_x_ref:
            # infer features from preprocessed reference data
            self.n_features = self.x_ref.reshape(self.x_ref.shape[0], -1).shape[-1]
        else:  # infer number of features after applying preprocessing step
            x = self.preprocess_fn(x_ref[0:1])
            self.n_features = x.reshape(x.shape[0], -1).shape[-1]

        if correction not in ['bonferroni', 'fdr'] and self.n_features > 1:
            raise ValueError('Only `bonferroni` and `fdr` are acceptable for multivariate correction.')

        # set metadata
        self.meta['detector_type'] = 'offline'  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = data_type

    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data preprocessing before computing the drift scores.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if isinstance(self.preprocess_fn, Callable):  # type: ignore[arg-type]
            x = self.preprocess_fn(x)
            x_ref = self.x_ref if self.preprocess_x_ref else self.preprocess_fn(self.x_ref)
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def feature_score(self, x_ref: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def score(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the feature-wise drift score which is the p-value of the
        statistical test and the test statistic.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and test statistics.
        """
        x_ref, x = self.preprocess(x)
        score, dist = self.feature_score(x_ref, x)  # feature-wise univariate test
        return score, dist

    def predict(self, x: Union[np.ndarray, list], drift_type: str = 'batch',
                return_p_val: bool = True, return_distance: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[np.ndarray, int, float]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        drift_type
            Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for
            each feature are aggregated using the Bonferroni or False Discovery Rate correction (if n_features>1).
        return_p_val
            Whether to return feature level p-values.
        return_distance
            Whether to return the test statistic between the features of the new batch and reference data.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the drift prediction and optionally the feature level p-values,
         threshold after multivariate correction if needed and test statistics.
        """
        # compute drift scores
        p_vals, dist = self.score(x)

        # TODO: return both feature-level and batch-level drift predictions by default
        # values below p-value threshold are drift
        if drift_type == 'feature':
            drift_pred = (p_vals < self.p_val).astype(int)
        elif drift_type == 'batch' and self.correction == 'bonferroni':
            threshold = self.p_val / self.n_features
            drift_pred = int((p_vals < threshold).any())  # type: ignore[assignment]
        elif drift_type == 'batch' and self.correction == 'fdr':
            drift_pred, threshold = fdr(p_vals, q_val=self.p_val)  # type: ignore[assignment]
        else:
            raise ValueError('`drift_type` needs to be either `feature` or `batch`.')

        # update reference dataset
        if isinstance(self.update_x_ref, dict) and self.preprocess_fn is not None and self.preprocess_x_ref:
            x = self.preprocess_fn(x)
        self.x_ref = update_reference(self.x_ref, x, self.n, self.update_x_ref)  # type: ignore[arg-type]
        # used for reservoir sampling
        self.n += len(x)

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        if return_p_val:
            cd['data']['p_val'] = p_vals
            cd['data']['threshold'] = self.p_val if drift_type == 'feature' else threshold
        if return_distance:
            cd['data']['distance'] = dist
        return cd
