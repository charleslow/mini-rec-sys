"""
Besides the default samplers, one can also subclass Sampler and define custom sampling logic.
"""
from typing import Optional, Sized
import torch
from mini_rec_sys.data.datasets import Dataset, SessionDataset
from torch import utils
from torch.utils.data import BatchSampler, Sampler
from pdb import set_trace


class SequentialSampler(Sampler):
    def __init__(self, data_source: SessionDataset) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return self.data_source.iterkeys()


class BatchedSequentialSampler(Sampler):
    """
    Returns batches of keys of the data_source based on iteration order.
    """

    def __init__(
        self, data_source: SessionDataset, batch_size: int, drop_last: bool = True
    ):
        self.sampler = BatchSampler(
            SequentialSampler(data_source),
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def __iter__(self):
        return self.sampler.__iter__()
