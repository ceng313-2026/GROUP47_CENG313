"""Helpers shared by the notebooks: plotting style, the competition metric,
score propagation and the submission writer."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import sparse

from .data import ASPECTS

# One fixed colour per entity, used in every figure.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ASPECT_COLORS = {"BPO": BLUE, "CCO": ORANGE, "MFO": AQUA}
SPLIT_COLORS = {"train": BLUE, "test": ORANGE}
NEUTRAL = "#52514e"
# Models get their own colours (plus line styles) so they are never confused with aspects.
VIOLET, YELLOW, MAGENTA = "#6250d6", "#eda100", "#e87ba4"
MODEL_STYLES = [(VIOLET, "-"), (YELLOW, "--"), (MAGENTA, ":")]


def set_plot_style() -> None:
    """Quiet, consistent matplotlib defaults (thin marks, light grid, no box)."""
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b8b7b2",
        "axes.labelcolor": NEUTRAL,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e7e6e1",
        "grid.linewidth": 0.8,
        "xtick.color": NEUTRAL,
        "ytick.color": NEUTRAL,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
    })


def format_count(n) -> str:
    """12345 -> '12,345'."""
    return f"{int(n):,}"


def propagate_scores(scores: np.ndarray, score_terms, go) -> np.ndarray:
    """Make predictions consistent with the GO hierarchy.

    A parent term must be at least as likely as any of its descendants, so each
    term receives the maximum score among itself and its descendants (within
    the predicted label set). Returns a new array.
    """
    index = {t: j for j, t in enumerate(score_terms)}
    descendants: dict[int, list[int]] = {}
    for term, j in index.items():
        for ancestor in go.ancestors(term):
            a = index.get(ancestor)
            if a is not None:
                descendants.setdefault(a, []).append(j)

    result = scores.copy()
    for a, cols in descendants.items():
        np.maximum(result[:, a], scores[:, cols].max(axis=1), out=result[:, a])
    return result


def write_submission(path, protein_ids, scores: np.ndarray, score_terms, top_k: int = 40,
                     min_score: float = 0.01, append: bool = False) -> int:
    """Write predictions in the CAFA format: `protein <TAB> GO term <TAB> score`.

    No header, scores in (0, 1] with three decimals. Only the `top_k` highest
    scoring terms of each protein that reach `min_score` are written.
    Returns the number of lines written.
    """
    scores = np.asarray(scores)
    n, n_terms = scores.shape
    k = min(top_k, n_terms)
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k < n_terms else np.tile(np.arange(n_terms), (n, 1))
    values = np.take_along_axis(scores, top, axis=1)

    frame = pd.DataFrame({
        "protein": np.repeat(np.asarray(protein_ids, dtype=object), k),
        "term": np.asarray(score_terms, dtype=object)[top.ravel()],
        "score": np.clip(values.ravel(), 0.0, 1.0),
    })
    frame = frame[frame["score"] >= max(min_score, 0.001)]
    frame = frame.sort_values(["protein", "score"], ascending=[True, False], kind="stable")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", header=False, index=False, float_format="%.3f",
                 mode="a" if append else "w")
    return len(frame)


# ---------------------------------------------------------------------------
# Competition metric
#
# CAFA evaluation: protein-centric, information-accretion weighted F-max.
#
# For one aspect (BPO, CCO or MFO) and one decision threshold t:
#
#     precision_i(t) = sum of IA over correctly predicted terms / sum of IA over predicted terms
#     recall_i(t)    = sum of IA over correctly predicted terms / sum of IA over true terms
#
# Precision is averaged over the proteins that have at least one prediction at
# threshold t, recall over all proteins that have ground truth in the aspect.
# F-max is the best F1 over all thresholds. The competition score is the mean
# of the three per-aspect F-max values.
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def weighted_fmax(y_true_full, true_weights, y_true_labels, y_score, label_weights,
                  thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Weighted F-max for one aspect.

    y_true_full   sparse (n, T)  ground truth over ALL terms of the aspect
    true_weights  (T,)           IA of those terms
    y_true_labels dense  (n, L)  ground truth restricted to the predicted terms
    y_score       dense  (n, L)  predicted scores in [0, 1]
    label_weights (L,)           IA of the predicted terms

    Recall uses every true term, including the ones the model cannot predict,
    so restricting the label space is correctly penalised.
    """
    true_weight_sum = np.asarray(y_true_full @ true_weights).ravel()
    has_truth = true_weight_sum > 0
    n_truth = int(has_truth.sum())
    curve = []
    if n_truth == 0 or y_score.shape[1] == 0:
        return {"fmax": 0.0, "threshold": np.nan, "precision": 0.0, "recall": 0.0,
                "n_proteins": n_truth, "curve": pd.DataFrame(columns=["threshold", "precision", "recall", "f"])}

    scores = np.asarray(y_score)[has_truth]
    truth = np.asarray(y_true_labels)[has_truth] > 0
    denom = true_weight_sum[has_truth]
    label_weights = np.asarray(label_weights, dtype=np.float64)

    best = {"fmax": 0.0, "threshold": np.nan, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        predicted = scores >= t
        pred_weight = predicted @ label_weights
        tp_weight = (predicted & truth) @ label_weights
        covered = pred_weight > 0
        precision = float(np.mean(tp_weight[covered] / pred_weight[covered])) if covered.any() else 0.0
        recall = float(np.mean(tp_weight / denom))
        f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        curve.append((float(t), precision, recall, f))
        if f > best["fmax"]:
            best = {"fmax": f, "threshold": float(t), "precision": precision, "recall": recall}

    best["n_proteins"] = n_truth
    best["curve"] = pd.DataFrame(curve, columns=["threshold", "precision", "recall", "f"])
    return best


def evaluate_predictions(y_true, true_terms, y_score, score_terms, ia, term_aspect,
                         thresholds=DEFAULT_THRESHOLDS, weighted: bool = True) -> dict:
    """Score predictions the way the competition does.

    y_true       sparse (n, T) ground truth, columns = `true_terms` (all terms)
    y_score      dense  (n, L) predictions, columns = `score_terms`
    ia           Series term -> information accretion
    term_aspect  dict   term -> 'BPO' / 'CCO' / 'MFO'
    weighted     False gives the plain (unweighted) F-max instead

    Returns {'BPO': {...}, 'CCO': {...}, 'MFO': {...}, 'mean_fmax': float}.
    """
    y_true = sparse.csc_matrix(y_true)
    true_terms = list(true_terms)
    score_terms = list(score_terms)
    true_index = {t: j for j, t in enumerate(true_terms)}

    def weights(term_list):
        if not weighted:
            return np.ones(len(term_list))
        return ia.reindex(term_list).fillna(0.0).to_numpy(dtype=np.float64)

    results = {}
    for aspect in ASPECTS:
        true_cols = [j for j, t in enumerate(true_terms) if term_aspect.get(t) == aspect]
        score_cols = [j for j, t in enumerate(score_terms) if term_aspect.get(t) == aspect]
        aspect_score_terms = [score_terms[j] for j in score_cols]

        # ground truth restricted to the predicted terms (all-zero column if unseen)
        y_true_labels = np.zeros((y_true.shape[0], len(score_cols)), dtype=bool)
        for k, term in enumerate(aspect_score_terms):
            j = true_index.get(term)
            if j is not None:
                column = y_true.getcol(j)
                y_true_labels[column.indices, k] = True

        results[aspect] = weighted_fmax(
            y_true[:, true_cols],
            weights([true_terms[j] for j in true_cols]),
            y_true_labels,
            np.asarray(y_score)[:, score_cols],
            weights(aspect_score_terms),
            thresholds=thresholds,
        )
    results["mean_fmax"] = float(np.mean([results[a]["fmax"] for a in ASPECTS]))
    return results


def results_row(name: str, results: dict) -> dict:
    """Flatten the output of `evaluate_predictions` into one table row."""
    row = {"model": name}
    for aspect in ASPECTS:
        row[f"Fmax {aspect}"] = results[aspect]["fmax"]
    row["Fmax mean"] = results["mean_fmax"]
    for aspect in ASPECTS:
        row[f"thr {aspect}"] = results[aspect]["threshold"]
    return row
