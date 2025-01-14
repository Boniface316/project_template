"""Base for high-level project jobs."""

# %%
import abc
import typing as T

import pydantic as pdt

from ..services import AlertsService, LoggerService, MLflowService
# %% TYPES

Locals = T.Dict[str, T.Any]


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for jobs.

    Use jobs to provide low-level preferences.
    i.e., to separate jobs from settings.

    Parameters:
        logger_service (LoggerService): manage the logger system.
        alerts_service (AlertsService): manage the alerts system.
        mlflow_service (MLflowService): manage the mlflow system.
    """

    KIND: str

    logger_service: LoggerService = LoggerService()
    alerts_service: AlertsService = AlertsService()
    mlflow_service: MLflowService = MLflowService()

    @abc.abstractmethod
    def run(self, locals: Locals) -> None:
        """Run the job."""
        raise NotImplementedError
