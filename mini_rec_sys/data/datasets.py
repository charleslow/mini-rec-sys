from diskcache import Cache
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from shutil import copytree
import torch

from mini_rec_sys.data.session import Session
from mini_rec_sys.constants import (
    ITEM_ATTRIBUTES_NAME,
    USER_ATTRIBUTES_NAME,
    SESSION_NAME,
)
from pdb import set_trace


class Dataset(torch.utils.data.Dataset):
    """
    A class used to load attributes associated with each user or item.
    At initialization, the attributes are stored in a simple database and retrieved
    during training or evaluation time.
    """

    def __init__(
        self,
        db_location: str = None,
        id_name: str = "id",
        load_fn: callable = None,
        store_fn: callable = None,
        data: dict | str = None,
        subset_ids: list[int | str] = None,
    ) -> None:
        """Initialize the Loader.

        If data is not provided, it will try to load a previously initialized db
        at db_location. If data is provided, it will write the values provided
        into db_location (or a temporary location if not provided).

        db_location: folder to store the db / where the db is stored
        id_name: the name of the id key for each user or item
        load_fn: given the object associated with each user / item, process the
            object for loading
        store_fn: given the (id, value) tuple for each row of raw data, preprocess
            the tuple with this function before storing the desired object for
            loading later on
        data: how to access the data
            if str, assumes that it is a file location containing .parquet or .json files.
            if dict, assumes that the key is the id and values are the attributes.
        subset_ids: Whether to limit this dataset to only a subset of keys as
            specified in the list of subset_ids.
        """
        assert not (
            data is None and db_location is None
        ), "Must provide db_location and/or data."
        self.db_location = db_location
        self.id_name = id_name

        if load_fn is None:
            load_fn = lambda x: x
        self.load_fn = load_fn

        if store_fn is None:
            store_fn = lambda id, row: row
        self.store_fn = store_fn

        if subset_ids is not None:
            self.subset_ids = set(subset_ids)
        else:
            self.subset_ids = subset_ids

        if data is not None:
            print(f"Populating database..")
            self.cache = self.populate_db(data)
        else:
            self.cache = Cache(db_location)
            print(
                f"Loading / initializing database with {len(self)} entries at {db_location}.."
            )

    def get_files_from_path(self, path: str, suffix="parquet"):
        """Get a list of .suffix files in path."""
        if path.endswith(suffix):
            return [path]
        files = Path(path).glob(f"*.{suffix}")
        return list(files)

    def populate_db(self, data: str | dict):
        # TODO: clean up temporary cache files.
        if isinstance(data, str):
            parquet_files = self.get_files_from_path(data, "parquet")
            num_parquet_files = len(parquet_files)
            json_files = self.get_files_from_path(data, "json")
            num_json_files = len(json_files)
            assert not (
                num_parquet_files > 0 and num_json_files > 0
            ), f"Should only have either .parquet or .json files in {data}."
            assert not (
                num_parquet_files == 0 and num_json_files == 0
            ), f"No .parquet or .json files found in {data}."
            if num_parquet_files > 0:
                generator = self.parquet_row_generator(parquet_files)
            if num_json_files > 0:
                generator = self.json_row_generator(json_files)

        elif isinstance(data, dict):
            generator = iter(data.items())

        else:
            raise ValueError(f"{data} is neither str nor dict.")

        if self.db_location is None:
            print("Initializing cache in temp location..")
            cache = Cache()
            self.db_location = cache.directory
        elif self.db_location.startswith("dbfs:/"):
            print("On databricks, writing to temp location..")
            cache = Cache()
        else:
            cache = Cache(self.db_location)

        for id, row in tqdm(generator):
            cache[id] = self.store_fn(id, row)

        if self.db_location and self.db_location.startswith("dbfs:/"):
            directory = cache.directory
            copytree(directory, self.db_location)
            cache = Cache(self.db_location)
        return cache

    def json_row_generator(self, files):
        for path in files:
            with open(path) as f:
                d = json.load(f)
            for id, values in d.items():
                yield id, values

    def parquet_row_generator(self, files):
        for path in files:
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                id = row.pop(self.id_name)
                yield id, row

    def load_object(self, id: int | str):
        """
        Load the raw object for id.
        """
        if self.subset_ids:
            if id not in self.subset_ids:
                return None
        return self.cache.get(id, None)

    def load(self, id: int | str):
        """
        Load the object for id, using load_fn to process it before returning.
        """
        object = self.load_object(id)
        if object is None:
            return None
        return self.load_fn(object)

    def __len__(self):
        if self.subset_ids:
            return len(self.subset_ids)
        return len(self.cache)

    def __getitem__(self, id: int | str):
        return self.load(id)

    def __iter__(self):
        self.keys = self.iterkeys()
        return self

    # For iterator style usage, we return both the key and result
    def __next__(self):
        k = next(self.keys)
        result = self.load(k)
        return k, result

    def iterkeys(self):
        if self.subset_ids:
            return iter(self.subset_ids)
        else:
            return self.cache.iterkeys()


# For UserDataset and ItemDataset, we just enforce that the load_fn must load
# a dict object
class UserItemDataset(Dataset):
    def __init__(
        self,
        db_location: str = None,
        id_name: str = "id",
        load_fn: callable = None,
        store_fn: callable = None,
        data: dict | str = None,
    ) -> None:
        super().__init__(db_location, id_name, load_fn, store_fn, data)
        self.check_returns_dict()

    def check_returns_dict(self, n=50):
        for i, v in enumerate(iter(self)):
            session_id, _ = v
            item = self.load(session_id)
            assert isinstance(item, dict), f"{type(self).__name__} must load dict."
            if i >= n:
                break


ItemDataset = UserItemDataset
UserDataset = UserItemDataset


class SessionDataset(Dataset):
    """
    Specific dataset for Sessions. Can specify user_dataset and item_dataset
    which will be used to load user and item attributes for each Session where
    applicable.

    Note that store_fn cannot be None, as it has to return a Session object.
    """

    def __init__(
        self,
        db_location: str = None,
        id_name: str = "id",
        load_fn: callable = None,
        store_fn: callable = None,
        data: dict | str = None,
        subset_ids: list = None,
        user_dataset: Dataset = None,
        item_dataset: Dataset = None,
    ) -> None:
        assert (
            store_fn is not None
        ), "SessionDataset must specify a store_fn that returns a Session."
        super().__init__(
            db_location=db_location,
            id_name=id_name,
            load_fn=load_fn,
            store_fn=store_fn,
            data=data,
            subset_ids=subset_ids,
        )
        self.user_dataset = user_dataset
        self.item_dataset = item_dataset
        self.check_returns_session()
        self.has_negative_items = self.check_negative_items()

    def check_returns_session(self, n=50):
        for i, v in enumerate(iter(self)):
            session_id, _ = v
            item = self.load(session_id)
            assert isinstance(item, Session), "SessionDataset must load Sessions."
            if i >= n:
                break

    def check_negative_items(self):
        v = next(iter(self))
        session: Session = v[1]
        return session.negative_items is not None

    def __getitem__(self, id: int | str):
        return self.load_session_dict(id)

    def load_session_dict(self, id: int | str):
        session: Session = self.load(id)

        if session is None:
            return None

        item_attributes = (
            session.items
            if self.item_dataset is None
            else self.load_items(session.items)
        )
        user_attributes = (
            session.user if self.user_dataset is None else self.load_users(session.user)
        )
        return {
            SESSION_NAME: session,
            USER_ATTRIBUTES_NAME: user_attributes,
            ITEM_ATTRIBUTES_NAME: item_attributes,
        }

    def load_item(self, item_id: int | str):
        res = {"item_id": item_id}
        if (
            self.item_dataset is None
            or (attrs := self.item_dataset.load(item_id)) is None
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
            self.user_dataset is None
            or (attrs := self.user_dataset.load(user_id)) is None
        ):
            return res
        res.update(attrs)
        return res

    def load_users(self, users: list[int | str] | int | str):
        if isinstance(users, list):
            return [self.load_user(user) for user in users]
        return self.load_user(users)

    def split_dataset(self, split_fn: callable):
        """
        Generates a new SessionDataset object that only contains a subset of
        keys from the parent, based on whether `split_fn(key)` is True.

        Note that it reuses the same cache as the parent, just that we
        restrict the keys for the child SessionDataset instance.
        """
        subset_ids = [k for k in self.iterkeys() if split_fn(k)]
        child = SessionDataset(
            db_location=self.db_location,
            id_name=self.id_name,
            load_fn=self.load_fn,
            store_fn=self.store_fn,
            item_dataset=self.item_dataset,
            user_dataset=self.user_dataset,
            subset_ids=subset_ids,
        )
        return child
