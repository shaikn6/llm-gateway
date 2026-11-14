"""Tune guarded cosine thresholds using stratified held-out pairs; entirely local."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("API_KEYS", "bench-key")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semantic-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.cache.embedding_cache import guard_keys

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / "data" / f"{name}.csv").open() as file:
        return list(csv.DictReader(file))


def scores(labels, predicted):
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / int(labels.sum()) if labels.sum() else 0.0,
        "false_positive_rate": fp / int((~labels).sum()) if (~labels).sum() else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "predicted_positive": tp + fp,
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
    }


def test_scores(labels, sims, allowed, test, threshold):
    return {
        key: scores(labels[test], (sims[test] >= threshold) & mask[test])
        for key, mask in [
            ("without_guards", np.ones(len(labels), dtype=bool)),
            ("with_guards", allowed),
        ]
    }


def split(labels):
    rng = np.random.default_rng(0)
    dev, test = [], []
    for label in [False, True]:
        ids = np.flatnonzero(labels == label)
        rng.shuffle(ids)
        n = len(ids) // 2 if not label else len(labels) // 2 - len(dev)
        dev.extend(ids[:n])
        test.extend(ids[n:])
    return np.array(dev), np.array(test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-vectors", action="store_true", help="Reuse locally derived embedding archives"
    )
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    embed = DefaultEmbeddingFunction()
    report = {
        "seed": 0,
        "split": "stratified 50/50 dev/test",
        "target_precision": 0.98,
        "datasets": {},
        "threshold_candidates": "0.001 grid plus every observed dev similarity; lowest qualifying candidate",
        "undefined_precision": "Reported as 0 when no positives are predicted",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    arrays = {}
    for ax, name in zip(axes, ["fintech_pairs", "paws_sample"]):
        pairs = rows(name)
        texts = list(dict.fromkeys(t for row in pairs for t in [row["text_a"], row["text_b"]]))
        vectors = {}
        vector_path = ROOT / "results" / f"{name}_vectors.npz"
        if args.reuse_vectors:
            archive = np.load(vector_path)
            vectors = dict(zip(archive["texts"], archive["vectors"]))
        else:
            for start in range(0, len(texts), 128):
                batch = texts[start : start + 128]
                vectors.update(zip(batch, embed(batch)))
                print(name, min(start + 128, len(texts)), "/", len(texts), flush=True)
        sims = np.array(
            [
                np.dot(vectors[r["text_a"]], vectors[r["text_b"]])
                / (np.linalg.norm(vectors[r["text_a"]]) * np.linalg.norm(vectors[r["text_b"]]))
                for r in pairs
            ]
        )
        labels = np.array([r["same_meaning"] == "1" for r in pairs])
        allowed = np.array([guard_keys(r["text_a"]) == guard_keys(r["text_b"]) for r in pairs])
        dev, test = split(labels)
        arrays[name] = sims, labels, allowed, test
        thresholds = np.unique(np.r_[np.linspace(0, 1, 1001), sims[dev]])
        choices = {}
        for guarded in [False, True]:
            mask = allowed if guarded else np.ones(len(pairs), dtype=bool)
            values = [scores(labels[dev], (sims[dev] >= t) & mask[dev]) for t in thresholds]
            qualifying = [i for i, v in enumerate(values) if v["precision"] >= 0.98]
            idx = (
                qualifying[0]
                if qualifying
                else max(range(len(values)), key=lambda i: values[i]["precision"])
            )
            key = "with_guards" if guarded else "without_guards"
            choices[key] = {
                "threshold": float(thresholds[idx]),
                "target_reached": bool(qualifying),
                "dev": values[idx],
                "test": scores(labels[test], (sims[test] >= thresholds[idx]) & mask[test]),
            }
            ax.plot(
                thresholds,
                [v["precision"] for v in values],
                label=f"precision {key}",
                linestyle="-" if guarded else "--",
            )
            ax.plot(
                thresholds,
                [v["recall"] for v in values],
                label=f"recall {key}",
                linestyle="-" if guarded else "--",
            )
        chosen = choices["with_guards"]["threshold"]
        report["datasets"][name] = {
            "pairs": len(pairs),
            "blocked": int((~allowed).sum()),
            "dev_pairs": len(dev),
            "test_pairs": len(test),
            "selection": choices,
            "test": test_scores(labels, sims, allowed, test, chosen),
        }
        # Save only derived vectors for a reproducible local benchmark, not in tracked results.
        np.savez_compressed(
            vector_path,
            texts=np.array(texts),
            vectors=np.array([vectors[t] for t in texts]),
        )
        ax.axvline(chosen, color="black", label=f"chosen {chosen:.6f}")
        ax.set(
            xlim=(0.5, 1.001),
            ylim=(0, 1.03),
            title=name,
            xlabel="cosine threshold",
            ylabel="dev score",
        )
        ax.legend(fontsize=6)
    report["threshold"] = max(
        d["selection"]["with_guards"]["threshold"] for d in report["datasets"].values()
    )
    report["threshold_policy"] = (
        "Maximum of per-dataset guarded dev thresholds. test uses each dataset guarded threshold; runtime_test uses the shared threshold."
    )
    # Re-evaluate both datasets at the single deployed threshold.
    for name, data in report["datasets"].items():
        sims, labels, allowed, test = arrays[name]
        # Preserve the report schema while reusing each dataset's computed arrays.
        values = test_scores(labels, sims, allowed, test, report["threshold"])
        data["runtime_test"] = {key: values[key] for key in ["with_guards", "without_guards"]}

    (ROOT / "results" / "threshold.json").write_text(json.dumps(report, indent=2) + "\n")
    for ax in axes:
        ax.axvline(report["threshold"], color="gray", linestyle=":", label="shared runtime")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "threshold.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
