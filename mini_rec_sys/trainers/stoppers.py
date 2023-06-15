from typing import Any
from mini_rec_sys.trainers.logger import Logger

class Stopper:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class StopByEpochs(Stopper):
    def __init__(self, n_epochs=1000) -> None:
        self.n_epochs = n_epochs

    def __call__(self, logger: Logger) -> bool:
        if len(logger.stats) == 0:
            return False
        return logger.epoch >= self.n_epochs


class StopByProgress(Stopper):
    def __init__(self, metric_col: str, k=10) -> None:
        self.metric_col = metric_col
        self.k = k

    def __call__(self, logger: Logger) -> bool:
        return logger.has_not_progressed(self.metric_col, self.k)
