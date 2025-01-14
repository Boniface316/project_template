from base import Job
import typing as T

from ..services import MLflowService
from ..io.csv import ReaderKind
from ..core.models import ModelKind, Model_
from ..core.metrics import MetricsKind, Metric_
import pydantic as pdt
from ..utils.splitters import SplitterKind, TrainTestSplitter
from ..registeries.savers import SaverKind, CustomSaver
from ..utils.signers import SignerKind, InferSigner
from ..registeries.registries import RegisterKind, MlflowRegister


class TrainingJob(Job):
    """Train and register a single AI/ML model.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model (models.ModelKind): machine learning model to train.
        metrics (metrics_.MetricsKind): metric list to compute.
        splitter (splitters.SplitterKind): data sets splitter.
        saver (SaverKind): model saver.
        signer (signers.SignerKind): model signer.
        registry (RegisterKind): model register.
    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Run
    run_config: MLflowService.RunConfig = MLflowService.RunConfig(name="Training")
    # Data
    inputs: ReaderKind = pdt.Field(..., discriminator="KINDS")
    targets: ReaderKind = pdt.Field(..., discriminator="KINDS")
    # Model
    models: ModelKind = pdt.Field(Model_(), discriminator="KINDS")
    # Metrics
    metrics: MetricsKind = [Metric_()]
    # Splitter
    splitter: SplitterKind = pdt.Field(TrainTestSplitter(), discriminator="KIND")
    # Saver
    saver: SaverKind = pdt.Field(CustomSaver(), discriminator="KIND")
    # Signer
    signer: SignerKind = pdt.Field(InferSigner(), discriminator="KIND")
    # Register
    registry: RegisterKind = pdt.Field(MlflowRegister(), discriminator="KIND")
