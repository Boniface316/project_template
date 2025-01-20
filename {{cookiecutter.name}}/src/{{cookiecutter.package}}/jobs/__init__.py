from .training import TrainingJob
from .tuning import TuningJob

JobKind = TrainingJob | TuningJob

__all__ = ["TrainingJob", "TuningJob", "JobKind"]
