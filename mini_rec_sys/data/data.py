from pydantic import Field, validator, BaseModel
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union
from mini_rec_sys.data.loaders import Loader
from pdb import set_trace


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


class SessionCollection(BaseModel):
    """A collection of sessions which forms a training or evaluation set.

    Besides the Session(s) themselves, the SessionCollection can also specify
    loaders which can be used to load more metadata about each User or Item within
    the Sessions, for e.g. each Item may have a description. These metadata will
    be made available to classes training or evaluating on the SessionCollection.
    """

    sessions: List[Session]
    item_loader: Optional[Loader] = None
    user_loader: Optional[Loader] = None

    class Config:
        arbitrary_types_allowed = True

    def load_item(self, item_id: int | str):
        res = {"item_id": item_id}
        if (
            self.item_loader is None
            or (attrs := self.item_loader.load(item_id)) is None
        ):
            return res
        res.update(attrs)
        return res

    def load_items(self, items: list[int | str] | int | str):
        if isinstance(items, list):
            return [self.load_item(item) for item in items]
        return self.load_item(items)

    def load_user(self, user_id: int | str):
        res = {"user_id": user_id}
        if self.user_loader is None:
            return res
        res.update(self.user_loader.load(user_id))
        return res

    def load_users(self, users: list[int | str] | int | str):
        if isinstance(users, list):
            return [self.load_user(user) for user in users]
        return self.load_user(users)
