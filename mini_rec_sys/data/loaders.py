from __future__ import annotations
from diskcache import Cache
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from pdb import set_trace
from shutil import copytree


class Loader:
    """
    A class used to load attributes associated with each user or item.
    At initialization, the attributes are stored in a simple database and retrieved
    during training or evaluation time.

    TODO: simple cache for faster loading.
    """

    def __init__(
        self,
        db_location: str = None,
        id_name: str = "id",
        load_fn: object = None,
        data: dict | str = None,
    ) -> None:
        """Initialize the Loader.

        If data is not provided, it will try to load a previously initialized db
        at db_location. If data is provided, it will write the values provided
        into db_location (or a temporary location if not provided).

        db_location: folder to store the db / where the db is stored
        id_name: the name of the id key for each user or item
        load_fn: given the object associated with each user / item, process the
            object for loading
        data: how to access the data
            if str, assumes that it is a file location containing .parquet or .json files.
            if dict, assumes that the key is the id and values are the attributes.
        """
        assert not (
            data is None and db_location is None
        ), "Must provide db_location and/or data."
        self.db_location = db_location
        self.id_name = id_name

        if load_fn is None:
            load_fn = lambda x: x
        self.load_fn = load_fn

        if data is not None:
            print(f"Populating database..")
            self.cache = self.populate_db(data)
        else:
            self.cache = Cache(db_location)
            print(
                f"Loading / initializing database with {len(self.cache)} entries at {db_location}.."
            )

    def populate_db(self, data: str | dict):
        # TODO: clean up temporary cache files.
        if isinstance(data, str):
            parquet_files = Path(data).glob("*.parquet")
            num_parquet_files = len(list(parquet_files))
            json_files = Path(data).glob("*.json")
            num_json_files = len(list(json_files))
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
        elif self.db_location.startswith("dbfs:/"):
            print("On databricks, writing to temp location..")
            cache = Cache()
        else:
            cache = Cache(self.db_location)

        for id, row in tqdm(generator):
            cache[id] = row

        if self.db_location.startswith("dbfs:/"):
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

    def load_object(self, id: str):
        """
        Load the raw object for id.
        """
        return self.cache.get(id, None)

    def load(self, id: str):
        """
        Load the object for id, using load_fn to process it before returning.
        """
        object = self.load_object(id)
        if object is None:
            return None
        return self.load_fn(object)
