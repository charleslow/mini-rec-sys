from pydantic import Field, validator
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Item:
    item_id: Union[int, str]
    description: Optional[str] = None


@dataclass
class User:
    user_id: Union[int, str]


@dataclass
class Session:
    """Session is a generalized concept to store relevance judgments for training and evaluation.

    Some examples of how we might define a session:
    - One user query and items that the user clicked on.
    - A triplet of (query, positive_item, negative_item)
    """

    session_id: Union[int, str]
    positive_items: List[Item] = Field(
        ...,
        description="Items that received a positive relevance signal in this session.",
    )
    relevances: List[Union[int, float]] = Field(
        ..., description="The relevance scores for each of the `positive_items`."
    )
    user: Optional[User] = None
    query: Optional[str] = None
    negative_items: Optional[List[Item]] = Field(
        None,
        description="Items that are deemed irrelevant in this session, e.g. implicit negatives based on impressions or explicit negatives",
    )

    @validator("relevances")
    def relevances_more_than_zero(cls, relevances):
        if not all([rel > 0 for rel in relevances]):
            raise AssertionError("All relevances must be more than zero!")
        return relevances

    @validator("relevances")
    def relevances_same_length_as_positive_items(cls, relevances, values):
        assert "positive_items" in values
        if len(relevances) != len(values["positive_items"]):
            raise AssertionError(
                "Length of relevances and positive_items must be the same!"
            )
        return relevances

    @validator("negative_items")
    def negative_positive_items_no_overlap(cls, negative_items, values):
        if len(set(negative_items).intersection(values["positive_items"])) > 0:
            raise AssertionError("negative_items and positive_items cannot overlap!")
        return negative_items
