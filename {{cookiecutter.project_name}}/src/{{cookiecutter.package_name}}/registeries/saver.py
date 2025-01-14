import abc
import typing as T
import pydantic as pdt
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext

from ..utils.signers import Signature
from ..core.schemas import Inputs, Outputs
from ..core.models import Model

from .registries import Info


class Saver(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for saving models in registry.

    Separate model definition from serialization.
    e.g., to switch between serialization flavors.

    Parameters:
        path (str): model path inside the Mlflow store.
    """

    KIND: str

    path: str = "model"

    @abc.abstractmethod
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        """Save a model in the model registry.

        Args:
            model (Model): project model to save.
            signature (Signature): model signature.
            input_example (Inputs): sample of inputs.

        Returns:
            Info: model saving information.
        """


class CustomSaver(Saver):
    """Saver for project models using the Mlflow PyFunc module.

    https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    """

    KIND: T.Literal["CustomSaver"] = "CustomSaver"

    class Adapter(PythonModel):  # type: ignore[misc]
        """Adapt a custom model to the Mlflow PyFunc flavor for saving operations.

        https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?#mlflow.pyfunc.PythonModel
        """

        def __init__(self, model: Model):
            """Initialize the custom saver adapter.

            Args:
                model (Model): project model.
            """
            self.model = model

        def predict(
            self,
            context: PythonModelContext,
            model_input: Inputs,
            params: dict[str, T.Any] | None = None,
        ) -> Outputs:
            """Generate predictions with a custom model for the given inputs.

            Args:
                context (mlflow.PythonModelContext): mlflow context.
                model_input (Inputs): inputs for the mlflow model.
                params (dict[str, T.Any] | None): additional parameters.

            Returns:
                Outputs: validated outputs of the project model.
            """
            return self.model.predict(inputs=model_input)

    @T.override
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        adapter = CustomSaver.Adapter(model=model)
        return mlflow.pyfunc.log_model(
            python_model=adapter,
            signature=signature,
            artifact_path=self.path,
            input_example=input_example,
        )


class BuiltinSaver(Saver):
    """Saver for built-in models using an Mlflow flavor module.

    https://mlflow.org/docs/latest/models.html#built-in-model-flavors

    Parameters:
        flavor (str): Mlflow flavor module to use for the serialization.
    """

    KIND: T.Literal["BuiltinSaver"] = "BuiltinSaver"

    flavor: str

    @T.override
    def save(
        self,
        model: Model,
        signature: Signature,
        input_example: Inputs,
    ) -> Info:
        builtin_model = model.get_internal_model()
        module = getattr(mlflow, self.flavor)
        return module.log_model(
            builtin_model,
            artifact_path=self.path,
            signature=signature,
            input_example=input_example,
        )


SaverKind = CustomSaver | BuiltinSaver
