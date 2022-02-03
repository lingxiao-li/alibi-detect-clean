from functools import partial
from typing import Callable, Type, Union
import numpy as np
import torch
import torch.nn as nn


def predict_batch(x: Union[list, np.ndarray, torch.Tensor], model: Union[Callable, nn.Module, nn.Sequential],
                  device: torch.device = None, batch_size: int = int(1e10), preprocess_fn: Callable = None,
                  dtype: Union[Type[np.generic], torch.dtype] = np.float32) -> Union[np.ndarray, torch.Tensor, tuple]:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        PyTorch model.
    device
        Device type used. The default None tries to use the GPU and falls back on CPU if needed.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    batch_size
        Batch size used during prediction.
    preprocess_fn
        Optional preprocessing function for each batch.
    dtype
        Model output type, e.g. np.float32 or torch.float32.

    Returns
    -------
    Numpy array, torch tensor or tuples of those with model outputs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    return_list = False
    preds = []  # type: Union[list, tuple]
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):  # type: ignore
                x_batch = preprocess_fn(x_batch)
            preds_tmp = model(x_batch.to(device))  # type: ignore
            if isinstance(preds_tmp, (list, tuple)):
                if len(preds) == 0:  # init tuple with lists to store predictions
                    preds = tuple([] for _ in range(len(preds_tmp)))
                    return_list = isinstance(preds_tmp, list)
                for j, p in enumerate(preds_tmp):
                    if device.type == 'cuda' and isinstance(p, torch.Tensor):
                        p = p.cpu()
                    preds[j].append(p if not return_np or isinstance(p, np.ndarray) else p.numpy())
            elif isinstance(preds_tmp, (np.ndarray, torch.Tensor)):
                if device.type == 'cuda' and isinstance(preds_tmp, torch.Tensor):
                    preds_tmp = preds_tmp.cpu()
                preds.append(preds_tmp if not return_np or isinstance(preds_tmp, np.ndarray)  # type: ignore
                             else preds_tmp.numpy())
            else:
                raise TypeError(f'Model output type {type(preds_tmp)} not supported. The model output '
                                f'type needs to be one of list, tuple, np.ndarray or torch.Tensor.')
    concat = partial(np.concatenate, axis=0) if return_np else partial(torch.cat, dim=0)  # type: ignore[arg-type]
    out = tuple(concat(p) for p in preds) if isinstance(preds, tuple) \
        else concat(preds)  # type: Union[tuple, np.ndarray, torch.Tensor]
    if return_list:
        out = list(out)  # type: ignore[assignment]
    return out  # TODO: update return type with list