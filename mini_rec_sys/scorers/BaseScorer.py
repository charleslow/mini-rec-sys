from __future__ import annotations
from typing import Any


class BaseScorer:
    """
    All scorers should inherit from this class. Each scorer only needs to be
    concerned with generating scores for a single row of data or batched rows
    of data (for efficiency).

    cols: the key values of the input data that this scorer will use to generate
        its scores. Specifying the columns will allow us to check whether a
        pipeline is valid (i.e. whether upstream models supply the correct
        inputs to downstream models).
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.score(*args, **kwargs)

    def score(self, input_data: dict | list[dict]):
        raise NotImplementedError()
