from mini_rec_sys.data.loaders import Loader, SessionLoader
from mini_rec_sys.data.samplers import Sampler, SimpleSampler


class SessionCollection:
    """A collection of sessions which forms a training or evaluation set.

    Besides the Session(s) themselves, the SessionCollection can also specify
    loaders which can be used to load more metadata about each User or Item within
    the Sessions, for e.g. each Item may have a description. These metadata will
    be made available to classes training or evaluating on the SessionCollection.
    """

    def __init__(
        self,
        session_loader: SessionLoader,
        item_loader: Loader = None,
        user_loader: Loader = None,
        batch_size: int = 10,
        sampler: Sampler = SimpleSampler,
    ) -> None:
        """
        session_loader: A Loader with key `session_id` and value Session.
        item_loader: If specified, item attributes will be returned when loading items.
        user_loader: If specified, user attributes will be returned when loading users.
        batch_size: How many sessions in a mini batch.
        sampler: Class constructor to define how we want to sample sessions.
        """
        self.session_loader = session_loader
        self.item_loader = item_loader
        self.user_loader = user_loader
        self.batch_size = batch_size
        self.sampler = sampler(session_loader, batch_size)

    def load(self):
        """
        Load a mini batch of sessions for training or evaluation.
        """
        session_ids = self.sampler.load()
        if len(session_ids) == 0:
            return None
        return [self.session_loader.load(session_id) for session_id in session_ids]

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
        if (
            self.user_loader is None
            or (attrs := self.user_loader.load(user_id)) is None
        ):
            return res
        res.update(attrs)
        return res

    def load_users(self, users: list[int | str] | int | str):
        if isinstance(users, list):
            return [self.load_user(user) for user in users]
        return self.load_user(users)
