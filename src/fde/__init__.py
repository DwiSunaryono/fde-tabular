"""FDE: Fast-Deployment Ensemble for imbalanced tabular classification."""
__version__ = "1.1.1"
__all__ = ["FDEClassifier", "speed_report", "__version__"]


def __getattr__(name):
    # FDEClassifier is imported on first use, so that the command line can print its first message before the
    # learner libraries load
    if name == "FDEClassifier":
        from .classifier import FDEClassifier
        return FDEClassifier
    if name == "speed_report":
        from .speed import speed_report
        return speed_report
    raise AttributeError(name)
