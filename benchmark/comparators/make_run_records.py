"""Builds the run records that the comparator scripts read, from the public files of this archive.

For each dataset and seed, two files are written to <out dir>:
  <dataset>_seed<s>.npz   train_idx, test_idx, y_train, y_test, from holdout(y, seed) of datasets.py, checked against
                          the hashes of results/splits.csv
  <dataset>_seed<s>.json  the fields that the scripts read: dataset, seed, K, gate_metric (the run metric, macroF1 or
                          wF1), identity.source and identity.data_id, and the time limit of results/runs.csv stored as
                          the single entry of info.full.p1_spec.tune_seconds, so that the sum and clipping in the
                          scripts return that limit
A file names.json with the datasets written is added for autosklearn_export_data.py. Datasets that cannot be loaded
here (Kaggle datasets not yet downloaded) are skipped and listed.
Usage (from this folder): python make_run_records.py <out dir> [--autogluon-default] [dataset ...]
  --autogluon-default  use autogluon_default_time_limit_s instead of automl_time_limit_s (see README)"""
import csv, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets import SPEC, holdout, load  # noqa: E402

args = sys.argv[1:]
out = args.pop(0)
col = "autogluon_default_time_limit_s" if "--autogluon-default" in args else "automl_time_limit_s"
names = [a for a in args if not a.startswith("--")]
runs = list(csv.DictReader(open(os.path.join(HERE, "..", "results", "runs.csv"))))
splits = {(r["dataset"], r["seed"]): r for r in csv.DictReader(open(os.path.join(HERE, "..", "results", "splits.csv")))}
names = names or sorted({r["dataset"] for r in runs})
sha = lambda a: hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()
os.makedirs(out, exist_ok=True)
written, skipped = [], []
for name in names:
    try:
        _, y = load(name)
    except Exception as e:                              # source not available on this machine
        skipped.append(f"{name} ({type(e).__name__})"); continue
    for r in (r for r in runs if r["dataset"] == name):
        s = int(r["seed"]); tr, te = holdout(y, s); tr, te = np.sort(tr), np.sort(te)
        h = splits[(name, r["seed"])]
        if (sha(tr), sha(te), sha(y[te])) != (h["train_rows_sha256"], h["test_rows_sha256"], h["test_labels_sha256"]):
            sys.exit(f"split of {name} seed {s} differs from results/splits.csv")
        b = os.path.join(out, f"{name}_seed{s}")
        np.savez_compressed(b + ".npz", train_idx=tr, test_idx=te, y_train=y[tr], y_test=y[te])
        e = SPEC[name]
        json.dump({"dataset": name, "seed": s, "K": int(len(np.unique(y))),
                   "gate_metric": {"macro_f1": "macroF1", "weighted_f1": "wF1"}[r["run_metric"]],
                   "identity": {"source": e["source"], "data_id": str(e.get("id", name))},
                   "info": {"full": {"p1_spec": {"tune_seconds": {"time_limit": [float(r[col])]}}}}}, open(b + ".json", "w"))
    written.append(name)
json.dump(written, open(os.path.join(out, "names.json"), "w"))
print(f"records for {len(written)} datasets in {out}, time limit from {col}")
for s in skipped:
    print("skipped:", s)
