"""
Besides the default samplers, one can also subclass Sampler and define custom sampling logic.
"""
import torch
from mini_rec_sys.data.datasets import Dataset, SessionDataset
from torch import utils
from torch.utils.data import SequentialSampler, BatchSampler, Sampler
from pdb import set_trace


class BatchedSequentialSampler(Sampler):
    def __init__(
        self, data_source: utils.data.Dataset, batch_size: int, drop_last: bool = True
    ):
        self.sampler = BatchSampler(
            SequentialSampler(data_source), batch_size=batch_size, drop_last=drop_last
        )

    def __iter__(self):
        return self.sampler.__iter__()


# class SimpleSampler(Sampler):
#    """
#    Simply iterate over the SessionLoader and load items in sequence.
#    """
#
#    def __init__(self, loader: SessionLoader, batch_size=32) -> None:
#        super().__init__(loader, batch_size)
#        self.keys = self.loader.cache.iterkeys()
#
#    def load(self):
#        keys = []
#        while len(keys) < self.batch_size:
#            try:
#                keys.append(next(self.keys))
#            except StopIteration:
#                break
#        return keys
#
