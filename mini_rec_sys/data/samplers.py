"""
Samplers take in a SessionLoader and specifies a `load` method which returns
a mini batch of session keys based on some sampling logic.
"""
from mini_rec_sys.data.loaders import Loader, SessionLoader
from pdb import set_trace


class Sampler:
    def __init__(self, loader: SessionLoader, batch_size: int = 32) -> None:
        self.loader = loader
        self.batch_size = batch_size

    def load(self):
        raise NotImplementedError()


class SimpleSampler(Sampler):
    """
    Simply iterate over the SessionLoader and load items in sequence.
    """

    def __init__(self, loader: SessionLoader, batch_size=32) -> None:
        super().__init__(loader, batch_size)
        self.keys = self.loader.cache.iterkeys()

    def load(self):
        keys = []
        while len(keys) < self.batch_size:
            try:
                keys.append(next(self.keys))
            except StopIteration:
                break
        return keys
