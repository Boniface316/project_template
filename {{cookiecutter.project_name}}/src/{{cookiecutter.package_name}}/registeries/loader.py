import abc
import pydantic as pdt

from ..core.schemas import Inputs, Outputs
from ..core.schemas import OutputsSchema
import mlflow
from mlflow.pyfunc import PyFuncModel

import typing as T


class Loader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for loading models from registry.

    Separate model definition from deserialization.
    e.g., to switch between deserialization flavors.
    """

    KIND: str

    class Adapter(abc.ABC):
        """Adapt any model for the project inference."""

        @abc.abstractmethod
        def predict(self, inputs: Inputs) -> Outputs:
            """Generate predictions with the internal model for the given inputs.

            Args:
                inputs (Inputs): validated inputs for the project model.

            Returns:
                Outputs: validated outputs of the project model.
            """

    @abc.abstractmethod
    def load(self, uri: str) -> "Loader.Adapter":
        """Load a model from the model registry.

        Args:
            uri (str): URI of a model to load.

        Returns:
            Loader.Adapter: model loaded.
        """


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


LoaderKind = CustomLoader | BuiltinLoader
