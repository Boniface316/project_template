import os
import typing as T
import pandas as pd
from .reader import Reader, Lineage
from .writer import Writer


class ExampleReader(Reader):
    """Read a dataframe from an example dataset."""

    KIND: T.Literal["ExampleReader"] = "ExampleReader"
    path: str
    limit: int | None = None

    def __init__(self):
        self.create_input()
        self.create_target()

    def create_input(self):
        inputs = {
            "index": range(10),
            "A": [float(i) + 0.5 for i in range(10)],
            "B": [i + 1 for i in range(10)],
            "C": [f"string_{i}" for i in range(10)],
            "D": [i * 2 for i in range(10)],
        }
        df = pd.DataFrame(inputs)
        df.set_index("index", inplace=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/input.csv")

    def create_target(self):
        targets = {
            "index": range(10),
            "target": [i % 2 for i in range(10)],
        }
        df = pd.DataFrame(targets)
        df.set_index("index", inplace=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/target.csv")

    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path)
        if self.limit is not None:
            data = data.head(self.limit)
        return data

    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        pass


class ExampleWriter(Writer):
    """Example writer for a dataset.

    Parameters:
        path (str): path to write the dataset.
    """

    KIND: T.Literal["ExampleWriter"] = "ExampleWriter"

    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """
        data.to_csv(self.path, index=False)
