from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from config import EVALUATION_TOLERANCES_SEC, THRESHOLD_VALUES, VALIDATION_TOLERANCE_SEC


def make_window_prediction_table(manifest: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    table = manifest.iloc[result["indices"]].reset_index(drop=True).copy()
    table["transition_probability"] = result["transition_probabilities"]
    table["predicted_offset"] = result["predicted_offsets"]
    duration = table["window_end_time"] - table["window_start_time"]
    table["predicted_transition_time"] = table["window_start_time"] + table["predicted_offset"] * duration
    return table


def aggregate_recording_predictions(window_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (subject, recording_key), group in window_table.groupby(["subject", "recording_key"], sort=False):
        best = group.iloc[int(np.argmax(group["transition_probability"].to_numpy(dtype=np.float64)))]
        max_prob = float(best["transition_probability"])
        pred_event = int(max_prob >= threshold)
        pred_time = float(best["predicted_transition_time"]) if pred_event else np.nan
        true_event = int(group["recording_has_transition"].iloc[0])
        true_time = float(group["true_transition_time"].iloc[0]) if true_event else np.nan
        error = pred_time - true_time if true_event and pred_event else np.nan
        rows.append({
            "subject": int(subject),
            "recording_key": recording_key,
            "recording_stem": group["recording_stem"].iloc[0],
            "true_has_transition": true_event,
            "predicted_has_transition": pred_event,
            "maximum_probability": max_prob,
            "true_transition_time": true_time,
            "predicted_transition_time": pred_time,
            "signed_timing_error_sec": error,
            "absolute_timing_error_sec": abs(error) if np.isfinite(error) else np.nan,
        })
    return pd.DataFrame(rows)


def tolerance_metrics(recording_table: pd.DataFrame, tolerance_sec: float) -> dict[str, Any]:
    tp = fp = fn = 0
    for _, row in recording_table.iterrows():
        true_event = bool(row["true_has_transition"])
        pred_event = bool(row["predicted_has_transition"])
        if true_event and pred_event:
            error = abs(float(row["predicted_transition_time"] - row["true_transition_time"]))
            if error <= tolerance_sec:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif pred_event and not true_event:
            fp += 1
        elif true_event and not pred_event:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_sec": float(tolerance_sec), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def evaluate_recordings(recording_table: pd.DataFrame) -> dict[str, Any]:
    y_true = recording_table["true_has_transition"].to_numpy(dtype=np.int64)
    y_pred = recording_table["predicted_has_transition"].to_numpy(dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    matched = recording_table[
        (recording_table["true_has_transition"] == 1)
        & (recording_table["predicted_has_transition"] == 1)
    ]
    abs_errors = matched["absolute_timing_error_sec"].dropna().to_numpy(dtype=np.float64)
    signed_errors = matched["signed_timing_error_sec"].dropna().to_numpy(dtype=np.float64)

    return {
        "presence_accuracy": float(accuracy_score(y_true, y_pred)),
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "presence_f1": float(f1),
        "matched_transition_count": int(len(abs_errors)),
        "timing_mae_sec": float(np.mean(abs_errors)) if len(abs_errors) else float("nan"),
        "timing_rmse_sec": float(np.sqrt(np.mean(signed_errors ** 2))) if len(signed_errors) else float("nan"),
        "timing_median_absolute_error_sec": float(np.median(abs_errors)) if len(abs_errors) else float("nan"),
        "tolerance_metrics": [tolerance_metrics(recording_table, x) for x in EVALUATION_TOLERANCES_SEC],
    }


def choose_threshold(window_table: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows, best_threshold = [], 0.50
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for threshold in THRESHOLD_VALUES:
        recordings = aggregate_recording_predictions(window_table, float(threshold))
        event = tolerance_metrics(recordings, VALIDATION_TOLERANCE_SEC)
        presence = evaluate_recordings(recordings)
        score = (
            event["f1"], presence["presence_f1"], presence["presence_accuracy"],
            -abs(float(threshold) - 0.50),
        )
        rows.append({
            "threshold": float(threshold),
            "event_f1": event["f1"],
            "presence_accuracy": presence["presence_accuracy"],
            "presence_f1": presence["presence_f1"],
        })
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold, pd.DataFrame(rows)


def save_terrain_outputs(
    output_dir: Path, split_name: str, manifest: pd.DataFrame,
    result: dict[str, Any], label_encoder,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = manifest.iloc[result["terrain_indices"]].reset_index(drop=True).copy()
    predictions["true_id"] = result["terrain_y_true"]
    predictions["pred_id"] = result["terrain_y_pred"]
    predictions["true_terrain"] = label_encoder.inverse_transform(result["terrain_y_true"])
    predictions["pred_terrain"] = label_encoder.inverse_transform(result["terrain_y_pred"])
    predictions.to_csv(output_dir / f"{split_name}_terrain_predictions.csv", index=False)

    label_ids = np.arange(len(label_encoder.classes_))
    report = classification_report(
        result["terrain_y_true"], result["terrain_y_pred"], labels=label_ids,
        target_names=label_encoder.classes_, zero_division=0, output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{split_name}_terrain_classification_report.csv")
    matrix = confusion_matrix(result["terrain_y_true"], result["terrain_y_pred"], labels=label_ids)
    pd.DataFrame(
        matrix,
        index=[f"true_{x}" for x in label_encoder.classes_],
        columns=[f"pred_{x}" for x in label_encoder.classes_],
    ).to_csv(output_dir / f"{split_name}_terrain_confusion_matrix.csv")
    return predictions


def save_transition_outputs(
    output_dir: Path, split_name: str, window_table: pd.DataFrame, threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    window_table.to_csv(output_dir / f"{split_name}_transition_window_predictions.csv", index=False)
    recordings = aggregate_recording_predictions(window_table, threshold)
    recordings.to_csv(output_dir / f"{split_name}_transition_recording_predictions.csv", index=False)
    metrics = evaluate_recordings(recordings)
    with open(output_dir / f"{split_name}_transition_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"probability_threshold": threshold, **metrics}, f, indent=2, allow_nan=True)
    return metrics, recordings
