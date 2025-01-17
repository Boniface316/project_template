from .splitter import TrainTestSplitter, TimeSeriesSplitter
from .example import ExampleSplitter

SplitterKind = TrainTestSplitter | TimeSeriesSplitter | ExampleSplitter

__all__ = ["TrainTestSplitter", "TimeSeriesSplitter", "ExampleSplitter"]
