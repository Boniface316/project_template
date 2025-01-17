"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T


from .base import Locals, Job

import pydantic as pdt
from ..io import ReaderKind

from ..io.splitters import SplitterKind
from ..io.splitters import ExampleSplitter as TrainTestSplitter
from ..io import schemas
from ..services import MlflowService

import os
import mlflow

from ..models import ModelKind, ExampleModel

# %% JOBS


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

    # # Run
    run_config: MlflowService.RunConfig = MlflowService.RunConfig(name="Training")
    # # Data
    inputs: ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: ReaderKind = pdt.Field(..., discriminator="KIND")
    # # Model
    model: ModelKind = pdt.Field(ExampleModel(), discriminator="KIND")
    # # Metrics
    # metrics: metrics_.MetricsKind = [metrics_.SklearnMetric()]
    # Splitter
    splitter: SplitterKind = pdt.Field(TrainTestSplitter(), discriminator="KIND")
    # # Saver
    # saver: registries.SaverKind = pdt.Field(registries.CustomSaver(), discriminator="KIND")
    # # Signer
    # signer: signers.SignerKind = pdt.Field(signers.InferSigner(), discriminator="KIND")
    # # Registrer
    # # - avoid shadowing pydantic `register` pydantic function
    # registry: registries.RegisterKind = pdt.Field(registries.MlflowRegister(), discriminator="KIND")

    @T.override
    def run(self) -> Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.add("TrainingJob_{time}.log")
        logger.info("With logger: {}", logger)
        # # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            experiment_id = run.info.experiment_id
            run_id = run.info.run_id
            # Get the URI of the current run
            run_uri = client.get_run(run_id).info.artifact_uri
            directory = os.path.join(run_uri, experiment_id, run_id)
            logger.add(os.path.join(directory, "TrainingJob_{time}.log"))
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
            logger.info("Read targets: {}", self.targets)
            targets_ = self.targets.read()  # unchecked!
            targets = schemas.TargetsSchema.check(targets_)
            logger.debug("- Targets shape: {}", targets.shape)
            # lineage
            # - inputs
            logger.info("Log lineage: inputs")
            inputs_lineage = self.inputs.lineage(data=inputs, name="inputs")
            mlflow.log_input(dataset=inputs_lineage, context=self.run_config.name)
            logger.debug("- Inputs lineage: {}", inputs_lineage.to_dict())
            # - targets
            logger.info("Log lineage: targets")
            targets_lineage = self.targets.lineage(
                data=targets, name="targets", targets=schemas.TargetsSchema.cnt
            )
            mlflow.log_input(dataset=targets_lineage, context=self.run_config.name)
            logger.debug("- Targets lineage: {}", targets_lineage.to_dict())
            # splitter
            logger.info("With splitter: {}", self.splitter)
            # - index
            train_index, test_index = next(
                self.splitter.split(inputs=inputs, targets=targets)
            )
            # - inputs
            inputs_train = T.cast(schemas.Inputs, inputs.iloc[train_index])
            inputs_test = T.cast(schemas.Inputs, inputs.iloc[test_index])
            logger.debug("- Inputs train shape: {}", inputs_train.shape)
            logger.debug("- Inputs test shape: {}", inputs_test.shape)
            # - targets
            targets_train = T.cast(schemas.Targets, targets.iloc[train_index])
            targets_test = T.cast(schemas.Targets, targets.iloc[test_index])
            logger.debug("- Targets train shape: {}", targets_train.shape)
            logger.debug("- Targets test shape: {}", targets_test.shape)
            # model
            logger.info("Fit model: {}", self.model)
            self.model.fit(inputs=inputs_train, targets=targets_train)
            # outputs
            logger.info("Predict outputs: {}", len(inputs_test))
            outputs_test = self.model.predict(inputs=inputs_test)
            logger.debug("- Outputs test shape: {}", outputs_test.shape)
            # metrics
            # for i, metric in enumerate(self.metrics, start=1):
            #     logger.info("{}. Compute metric: {}", i, metric)
            #     score = metric.score(targets=targets_test, outputs=outputs_test)
            #     client.log_metric(run_id=run.info.run_id, key=metric.name, value=score)
            #     logger.debug("- Metric score: {}", score)
            # signer
            # logger.info("Sign model: {}", self.signer)
            # model_signature = self.signer.sign(inputs=inputs, outputs=outputs_test)
            # logger.debug("- Model signature: {}", model_signature.to_dict())
            # saver
            # logger.info("Save model: {}", self.saver)
            # model_info = self.saver.save(
            #     model=self.model, signature=model_signature, input_example=inputs
            # )
            # logger.debug("- Model URI: {}", model_info.model_uri)
            # register
            # logger.info("Register model: {}", self.registry)
            # model_version = self.registry.register(
            #     name=self.mlflow_service.registry_name, model_uri=model_info.model_uri
            # )
            # logger.debug("- Model version: {}", model_version)
            # notify
            # self.alerts_service.notify(
            #     title="Training Job Finished",
            #     message=f"Model version: {model_version.version}",
            # )
        return locals()
