from .example import ExampleMetric
import typing as T
import pydantic as pdt
from .base import Metric

MetricKind = ExampleMetric
MetricsKind: T.TypeAlias = list[
    T.Annotated[MetricKind, pdt.Field(discriminator="KIND")]
]

__all__ = ["MetricKind", "MetricsKind", "Metric"]