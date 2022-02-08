import numpy as np
from typing import Callable, Dict, Optional, Union
from .base import BaseUnivariateDrift


class ChiSquareDrift(BaseUnivariateDrift):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            categories_per_feature: Optional[Dict[int, int]] = None,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            correction: str = 'bonferroni',
            n_features: Optional[int] = None,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            correction=correction,
            n_features=n_features,
            input_shape=input_shape,
            data_type=data_type
        )
        # construct categories from the user-specified dict
        if isinstance(categories_per_feature, dict):
            vals = list(categories_per_feature.values())
            int_types = (int, np.int16, np.int32, np.int64)
            if all(isinstance(v, int_types) for v in vals):
                # categories_per_feature = Dict[int, int]
                categories_per_feature = {f: list(np.arange(v))  # type: ignore
                                          for f, v in categories_per_feature.items()}
            elif not all(isinstance(val, list) for val in vals) and \
                    all(isinstance(v, int_types) for val in vals for v in val):  # type: ignore
                raise ValueError('categories_per_feature needs to be None or one of '
                                 'Dict[int, int], Dict[int, List[int]]')
        else:  # infer number of possible categories for each feature from reference data
            x_flat = self.x_ref.reshape(self.x_ref.shape[0], -1)
            categories_per_feature = {f: list(np.unique(x_flat[:, f]))  # type: ignore
                                      for f in range(self.n_features)}
        self.x_ref_categories = categories_per_feature
