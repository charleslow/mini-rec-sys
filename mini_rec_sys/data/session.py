from pydantic import Field, validator
from pydantic.dataclasses import dataclass
from typing import Optional


@dataclass
class Session:
    """Session is a generalized concept to store relevance judgments for training and evaluation.

    Some examples of how we might define a session:
    - One user query and items that the user clicked on.
    - A triplet of (query, positive_item, negative_item)
    """

    session_id: int | str
    positive_items: list[int | str] = Field(
        ...,
        description="Items that received a positive relevance signal in this session.",
    )
    positive_relevances: list[int | float] = Field(
        ..., description="The relevance scores for each of the `positive_items`."
    )
    user: Optional[int | str] = None
    query: Optional[str] = None
    negative_items: Optional[list[int | str]] = Field(
        None,
        description="Items that are deemed irrelevant in this session, e.g. implicit negatives based on impressions or explicit negatives",
    )

    @validator("positive_relevances")
    def at_least_one_relevance(cls, relevances):
        if relevances is None or len(relevances) == 0:
            raise AssertionError("must have at least one relevance!")
        return relevances

    @validator("positive_relevances")
    def relevances_more_than_zero(cls, relevances):
        if not all([rel > 0 for rel in relevances]):
            raise AssertionError("All relevances must be more than zero!")
        return relevances

    @validator("positive_relevances")
    def relevances_same_length_as_positive_items(cls, relevances, values):
        assert "positive_items" in values
        if len(relevances) != len(values["positive_items"]):
            raise AssertionError(
                "Length of positive_relevances and positive_items must be the same!"
            )
        return relevances

    @validator("negative_items")
    def negative_positive_items_no_overlap(cls, negative_items, values):
        if len(set(negative_items).intersection(values["positive_items"])) > 0:
            raise AssertionError("negative_items and positive_items cannot overlap!")
        return negative_items

    def __post_init_post_parse__(self):
        negative_items = [] if self.negative_items is None else self.negative_items
        self.items = negative_items + self.positive_items
        relevance_dict = {
            item: rel
            for item, rel in zip(self.positive_items, self.positive_relevances)
        }
        self.relevances = [relevance_dict.get(item, 0) for item in self.items]
