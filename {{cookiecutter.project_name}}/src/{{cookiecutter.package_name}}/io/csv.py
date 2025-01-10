"""This is an example of csv reader"""

from .reader import Reader
import typing as T
import pandas as pd


class CSVReader(Reader):
    """Read a dataframe from a parquet file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["CSVReader"] = "CSVReader"
    path: str

    @T.override
    def read(self):
        return pd.read_csv(self.path, nrows=self.limit)

    @T.override
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ):
        return super().lineage(name, data, targets, predictions)


ReaderKind = CSVReader
