"""Stand-in for the study loader read by the comparator scripts: load(name, source, data_id) returns the dataset as
datasets.py of this archive gives it, which equals the study loader on all 53 datasets. The arguments source and
data_id keep the call signature of the scripts and are not needed here."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import load as _load  # noqa: E402


def load(name, source=None, data_id=None):
    return _load(name)
