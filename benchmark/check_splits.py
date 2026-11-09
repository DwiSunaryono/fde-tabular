"""Checks that holdout(y, seed) of datasets.py regenerates the TRAIN/TEST split of each stored run.

For every dataset that can be loaded here, the sorted row indices of TRAIN and TEST, and the TEST labels in row order,
are hashed with SHA-256 and compared with results/splits.csv. Datasets that are not available (for example Kaggle
datasets not yet downloaded) are skipped and listed.
Usage: python check_splits.py [dataset ...]"""
import csv, hashlib, os, sys

import numpy as np

from datasets import holdout, load

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


rows = list(csv.DictReader(open(os.path.join(HERE, "results", "splits.csv"))))
names = sys.argv[1:] or sorted({r["dataset"] for r in rows})
same = differ = 0; skipped = []
for name in names:
    try:
        _, y = load(name)
    except Exception as e:                              # source not available on this machine
        skipped.append(f"{name} ({type(e).__name__})"); continue
    for r in (r for r in rows if r["dataset"] == name):
        tr, te = holdout(y, int(r["seed"]))
        tr, te = np.sort(tr), np.sort(te)
        ok = (sha(tr), sha(te), sha(y[te])) == (r["train_rows_sha256"], r["test_rows_sha256"], r["test_labels_sha256"])
        same += ok; differ += not ok
        if not ok:
            print("differs:", name, "seed", r["seed"])
print(f"{same} runs identical, {differ} differ, {len(skipped)} datasets skipped")
for s in skipped:
    print("skipped:", s)
