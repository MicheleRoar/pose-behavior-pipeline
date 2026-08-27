"""
segmentation/classifier/train_overlap_classifier.py
=====================================================
Step 2: fits a small logistic-regression classifier (2 features +
bias -- centroid_dist_norm, pixel_gap_px, matching what
overlap_resolution._resolve_overlap_merges computes) on labeled rows
from extract_overlap_candidates.py, and writes a weights JSON that
merge_fragments.py's --overlap-classifier can load in place of the two
fixed thresholds.

Gradient descent on standardized features for stable convergence with
small samples, then folded back into weights that operate on raw
feature values (so the output JSON needs no standardization at
inference). Refuses to train below --min-examples (default 15) or with
only one labeled class -- a fit on too little/one-sided data is worse
than the fixed thresholds it would replace.

Label column: 1/"true"/"same_body" = merge, 0/"false"/"different_people"
= keep separate, blank = unlabeled (skipped).

Usage:
    python -m segmentation.classifier.train_overlap_classifier \
        --labels overlap_candidates_9_group_1_3.csv \
                 overlap_candidates_9_individual_58.csv \
                 overlap_candidates_10_58.csv \
        --out overlap_classifier.json

Then:
    python -m segmentation.tools.run_osnet_window ... \
        --overlap-classifier overlap_classifier.json

Not runnable in this project's sandbox in the sense that it needs real
labeled CSVs from Michele's machine -- but has no psifx/torch
dependency itself (pure csv + numpy), so it can also just be run here
against a copy of the CSVs if that's ever more convenient.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FEATURES = ["centroid_dist_norm", "pixel_gap_px"]

_TRUE_LABELS = {"1", "true", "same_body", "yes"}
_FALSE_LABELS = {"0", "false", "different_people", "no"}


def _load_labeled_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_label = (row.get("label") or "").strip().lower()
                if raw_label == "":
                    continue
                if raw_label in _TRUE_LABELS:
                    label = 1
                elif raw_label in _FALSE_LABELS:
                    label = 0
                else:
                    print(f"  skipping row (session={row.get('session')}, "
                          f"id_a={row.get('id_a')}, id_b={row.get('id_b')}): "
                          f"unrecognized label {raw_label!r}")
                    continue
                rows.append({
                    "session": row.get("session", Path(path).stem),
                    "id_a": row.get("id_a"),
                    "id_b": row.get("id_b"),
                    "centroid_dist_norm": float(row["median_centroid_dist_norm"]),
                    "pixel_gap_px": float(row["median_pixel_gap_px"]),
                    "label": label,
                })
    return rows


def _fit_logistic_regression(
    X: np.ndarray, y: np.ndarray, l2: float, lr: float = 0.5, n_iter: int = 20_000,
) -> tuple[np.ndarray, float]:
    """Batch gradient descent on standardized features. Returns
    `(weights, bias)` already converted back to operate on RAW
    (unstandardized) feature values -- see module docstring."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # guard a constant feature (e.g. all-zero pixel gap so far)
    Z = (X - mu) / sigma

    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        logits = Z @ w + b
        preds = 1.0 / (1.0 + np.exp(-logits))
        error = preds - y
        grad_w = (Z.T @ error) / n + l2 * w
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    # fold standardization back into raw-feature weights:
    # w_std . (x - mu) / sigma + b_std  ==  (w_std / sigma) . x + (b_std - w_std . mu / sigma)
    raw_w = w / sigma
    raw_b = b - float(w @ (mu / sigma))
    return raw_w, raw_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True,
                         help="One or more CSVs from extract_overlap_candidates.py, with the "
                              "'label' column filled in (blank rows are skipped)")
    parser.add_argument("--out", required=True, help="Output weights JSON path")
    parser.add_argument("--min-examples", type=int, default=15,
                         help="Refuse to train below this many total labeled rows")
    parser.add_argument("--l2", type=float, default=0.1, help="L2 regularization strength")
    args = parser.parse_args()

    rows = _load_labeled_rows(args.labels)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"{len(rows)} labeled example(s) loaded: {n_pos} same-body, {n_neg} different-people")

    if len(rows) < args.min_examples:
        raise SystemExit(
            f"Only {len(rows)} labeled example(s), need at least {args.min_examples} "
            f"(--min-examples) -- keep labeling more candidate pairs before training. "
            f"A fit on too little data is worse than the fixed thresholds it would replace."
        )
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            f"Need at least one example of BOTH classes to fit anything meaningful "
            f"(got {n_pos} same-body, {n_neg} different-people)."
        )

    X = np.array([[r["centroid_dist_norm"], r["pixel_gap_px"]] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=float)

    weights, bias = _fit_logistic_regression(X, y, l2=args.l2)

    # sanity check: report training accuracy at the 0.5 cutoff the
    # classifier will actually use at inference time (see
    # _resolve_overlap_merges) -- not a held-out estimate, just a
    # "did this even fit the data it was given" check.
    logits = X @ weights + bias
    preds = (1.0 / (1.0 + np.exp(-logits))) >= 0.5
    accuracy = float((preds == (y == 1)).mean())

    sessions = sorted({r["session"] for r in rows})
    out = {
        "features": FEATURES,
        "weights": [round(float(w), 6) for w in weights],
        "bias": round(float(bias), 6),
        "trained_on_n": len(rows),
        "trained_on_positive": n_pos,
        "trained_on_negative": n_neg,
        "trained_on_sessions": sessions,
        "train_accuracy": round(accuracy, 3),
        "l2": args.l2,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nweights: {dict(zip(FEATURES, out['weights']))}  bias: {out['bias']}")
    print(f"train accuracy (same data used to fit -- not a held-out estimate): {accuracy:.3f}")
    print(f"trained on {len(sessions)} session(s): {sessions}")
    print(f"\nwritten to {args.out}")
    print("Use with: --overlap-classifier " + args.out)


if __name__ == "__main__":
    main()
