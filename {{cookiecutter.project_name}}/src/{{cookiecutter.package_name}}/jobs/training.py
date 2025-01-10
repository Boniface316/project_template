from base import Job
import typing as T

from ..services import MLflowService
from ..io.csv import ReaderKind

import pydantic as pdt


class TrainingJob(Job):
    """Train and register a single AI/ML model.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model (models.ModelKind): machine learning model to train.
        metrics (metrics_.MetricsKind): metric list to compute.
        splitter (splitters.SplitterKind): data sets splitter.
        saver (registries.SaverKind): model saver.
        signer (signers.SignerKind): model signer.
        registry (registries.RegisterKind): model register.
    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Run
    run_config: MLflowService.RunConfig = MLflowService.RunConfig(name="Training")
    # Data
    inputs: ReaderKind = pdt.Field(..., discriminator="KINDS")
    targets: ReaderKind = pdt.Field(..., discriminator="KINDS")
    # Model
    models: ModelKind = pdt.Field(BaselineSklearnModel(), discriminator="KINDS")
