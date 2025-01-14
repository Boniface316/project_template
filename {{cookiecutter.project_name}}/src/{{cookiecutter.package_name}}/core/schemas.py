"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd

# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    """Schema for the inputs dataframe.

    Parameters:
        feature_1 (pa.Float): feature 1.
        feature_2 (pa.Float): feature 2.
        feature_3 (pa.Float): feature 3.
    """

    feature_1: pa.Float
    feature_2: pa.Float
    feature_3: pa.Float


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the targets dataframe.

    Parameters:
        target (pa.Float): target.
    """

    target: pa.Float


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the outputs dataframe.

    Parameters:
        prediction (pa.Float): prediction.
    """

    prediction: pa.Float


Outputs = papd.DataFrame[OutputsSchema]


class FeatureImportancesSchema(Schema):
    """Schema for the feature importances dataframe.

    Parameters:
        feature (pa.String): feature name.
        importance (pa.Float): feature importance.
    """

    feature: pa.String
    importance: pa.Float


FeatureImportances = papd.DataFrame[FeatureImportancesSchema]


class SHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


SHAPValues = papd.DataFrame[SHAPValuesSchema]
