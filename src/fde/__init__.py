"""FDE: Feasible Diverse Ensembles for tabular classification."""
__version__ = "1.0.1"
__all__ = ["FDEClassifier", "__version__"]


def __getattr__(name):
    # FDEClassifier is imported on first use, so that the command line can print its first message before the
    # learner libraries load
    if name == "FDEClassifier":
        from .classifier import FDEClassifier
        return FDEClassifier
    raise AttributeError(name)
