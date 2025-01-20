import typing as T

import mlflow
from mlflow.pyfunc import PyFuncModel

from ..io.schemas import Inputs, Outputs, OutputsSchema

from .base import Loader


class CustomLoader(Loader):
    """Loader for custom models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["CustomLoader"] = "CustomLoader"

    class Adapter(Loader.Adapter):
        """Adapt a custom model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: Inputs) -> Outputs:
            # model validation is already done in predict
            outputs = self.model.predict(data=inputs)
            return T.cast(Outputs, outputs)

    @T.override
    def load(self, uri: str) -> "CustomLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = CustomLoader.Adapter(model=model)
        return adapter


class BuiltinLoader(Loader):
    """Loader for built-in models using the Mlflow PyFunc module.

    Note: use Mlflow PyFunc instead of flavors to use standard API.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors
    """

    KIND: T.Literal["BuiltinLoader"] = "BuiltinLoader"

    class Adapter(Loader.Adapter):
        """Adapt a builtin model for the project inference."""

        def __init__(self, model: PyFuncModel) -> None:
            """Initialize the adapter from an mlflow pyfunc model.

            Args:
                model (PyFuncModel): mlflow pyfunc model.
            """
            self.model = model

        @T.override
        def predict(self, inputs: Inputs) -> Outputs:
            columns = list(OutputsSchema.to_schema().columns)
            outputs = self.model.predict(data=inputs)  # unchecked data!
            return Outputs(outputs, columns=columns, index=inputs.index)

    @T.override
    def load(self, uri: str) -> "BuiltinLoader.Adapter":
        model = mlflow.pyfunc.load_model(model_uri=uri)
        adapter = BuiltinLoader.Adapter(model=model)
        return adapter
