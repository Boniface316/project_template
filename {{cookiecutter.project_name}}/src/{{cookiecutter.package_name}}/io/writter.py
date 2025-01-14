import abc
import pydantic as pdt
import pandas as pd
import typing as T


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    e.g., to write file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Writer a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str

    @T.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_parquet(data, self.path)


class CSVWriter(Writer):
    """Writer a dataframe to a CSV file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["CSVWriter"] = "CSVWriter"

    path: str

    @T.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_csv(data, self.path)


WriterKind = ParquetWriter | CSVWriter
