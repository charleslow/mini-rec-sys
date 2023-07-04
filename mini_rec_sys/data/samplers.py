"""
Besides the default samplers, one can also subclass Sampler and define custom sampling logic.
"""
from __future__ import annotations
from typing import Optional, Sized
import torch
import random
import math
from mini_rec_sys.data.datasets import Dataset, SessionDataset
from mini_rec_sys.utils import flatten_list
from mini_rec_sys.constants import SESSION_WEIGHT_NAME
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


class BatchedWeightedSampler(Sampler):
    """
    Returns batches of keys of the data_source based on random sampling from
    the data_source keys based on the specified weights for each session.

    This can be useful for e.g. when our training dataset stores relevance
    judgments for each query as a Session, and we want to sample queries according
    to their occurrence frequency.
    """

    def __init__(
        self,
        data_source: SessionDataset,
        batch_size: int,
        drop_last: bool = True,
        num_instances: int = None,
    ):
        self.sampler = BatchSampler(
            WeightedSampler(
                data_source=data_source,
                num_instances=num_instances,
            ),
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def __iter__(self):
        return self.sampler.__iter__()


class WeightedSampler(Sampler):
    def __init__(
        self,
        data_source: SessionDataset,
        num_instances: int = None,
    ):
        """
        num_instances: if provided, will try to adjust the weights such that the
            sum of weights across all items is equal to num_instances. This is
            useful for e.g. when the weights stored are sample probabilities,
            and we want to sample without replacement from them.
        """
        print("Initializing WeightedSampler..")
        multiplier = 1.0
        if num_instances is not None:
            total_weight = sum(
                [data_source[k][SESSION_WEIGHT_NAME] for k in data_source.iterkeys()]
            )
            multiplier = num_instances / total_weight

        keys = []
        weights = []
        for k in data_source.iterkeys():
            weight = round(data_source[k][SESSION_WEIGHT_NAME] * multiplier)
            if weight > 0.0:
                keys.append(k)
                weights.append(int(weight))
        assert (
            len(keys) > 0
        ), "No keys with non-zero weight found, try increasing num_instances."
        self.keys = flatten_list([[k] * w for k, w in zip(keys, weights)])
        random.shuffle(self.keys)
        print(f"WeightedSampler initialized with {len(self.keys):,} instances.")

    def __iter__(self):
        return iter(self.keys)
