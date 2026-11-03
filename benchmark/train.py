from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE, SUBJECT_IDS, VALID_TERRAINS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    RANDOM_SEED, TERRAIN_LOSS_WEIGHT, TRANSITION_LOSS_WEIGHT, REGRESSION_LOSS_WEIGHT,
    VALIDATION_TOLERANCE_SEC, BaselineSpec,
)
from data import MultiTaskDataset, fit_scaler, load_manifest
from models import build_model
from evaluation import (
    aggregate_recording_predictions, choose_threshold, evaluate_recordings,
    make_window_prediction_table, save_terrain_outputs, save_transition_outputs,
    tolerance_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=DEVICE.type == "cuda",
        persistent_workers=num_workers > 0, drop_last=False,
    )


def class_weights(train_df: pd.DataFrame, label_encoder: LabelEncoder) -> torch.Tensor:
    clean = train_df[train_df["terrain_valid"] == 1]
    labels = label_encoder.transform(clean["terrain"].astype(str))
    counts = np.bincount(labels, minlength=len(label_encoder.classes_)).astype(np.float64)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def positive_weight(train_df: pd.DataFrame) -> torch.Tensor:
    labels = train_df["transition_label"].to_numpy(dtype=np.float32)
    positive = float(np.sum(labels == 1.0))
    negative = float(np.sum(labels == 0.0))
    if positive < 1 or negative < 1:
        raise ValueError("Training fold must contain positive and negative transition windows.")
    return torch.tensor(negative / positive, dtype=torch.float32, device=DEVICE)


def run_epoch(model, loader, terrain_criterion, pos_weight, optimizer=None, amp_scaler=None) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    use_amp = DEVICE.type == "cuda"

    total_loss = terrain_loss_sum = transition_loss_sum = regression_loss_sum = 0.0
    terrain_count = 0
    terrain_true, terrain_pred, terrain_indices = [], [], []
    trans_labels, trans_probs, offsets, all_indices = [], [], [], []

    for imu, frames, terrain_labels, terrain_valid, transition_labels, transition_offsets, indices in loader:
        imu = imu.to(DEVICE, non_blocking=True)
        frames = frames.to(DEVICE, non_blocking=True) if frames.numel() else None
        terrain_labels = terrain_labels.to(DEVICE, non_blocking=True)
        terrain_valid = terrain_valid.to(DEVICE, non_blocking=True)
        transition_labels = transition_labels.to(DEVICE, non_blocking=True)
        transition_offsets = transition_offsets.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                terrain_logits, transition_logits, predicted_offsets = model(imu, frames)
                terrain_loss = (
                    terrain_criterion(terrain_logits[terrain_valid], terrain_labels[terrain_valid])
                    if terrain_valid.any() else terrain_logits.sum() * 0.0
                )
                transition_loss = F.binary_cross_entropy_with_logits(
                    transition_logits, transition_labels, pos_weight=pos_weight
                )
                positive = transition_labels > 0.5
                regression_loss = (
                    F.smooth_l1_loss(predicted_offsets[positive], transition_offsets[positive])
                    if positive.any() else transition_logits.sum() * 0.0
                )
                loss = (
                    TERRAIN_LOSS_WEIGHT * terrain_loss
                    + TRANSITION_LOSS_WEIGHT * transition_loss
                    + REGRESSION_LOSS_WEIGHT * regression_loss
                )

            if training:
                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

        batch = transition_labels.size(0)
        total_loss += loss.item() * batch
        transition_loss_sum += transition_loss.item() * batch
        regression_loss_sum += regression_loss.item() * batch

        valid_count = int(terrain_valid.sum().item())
        if valid_count:
            terrain_count += valid_count
            terrain_loss_sum += terrain_loss.item() * valid_count
            preds = torch.argmax(terrain_logits[terrain_valid], dim=1)
            terrain_true.extend(terrain_labels[terrain_valid].detach().cpu().tolist())
            terrain_pred.extend(preds.detach().cpu().tolist())
            terrain_indices.extend(indices[terrain_valid.cpu()].tolist())

        probabilities = torch.sigmoid(transition_logits)
        trans_labels.extend(transition_labels.detach().cpu().numpy().tolist())
        trans_probs.extend(probabilities.detach().cpu().numpy().tolist())
        offsets.extend(predicted_offsets.detach().cpu().numpy().tolist())
        all_indices.extend(indices.tolist())

    terrain_true = np.asarray(terrain_true, dtype=np.int64)
    terrain_pred = np.asarray(terrain_pred, dtype=np.int64)
    labels = np.asarray(trans_labels, dtype=np.float32)
    probabilities = np.asarray(trans_probs, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    binary = (probabilities >= 0.5).astype(np.int64)

    result = {
        "loss": total_loss / len(loader.dataset),
        "terrain_loss": terrain_loss_sum / terrain_count if terrain_count else float("nan"),
        "transition_loss": transition_loss_sum / len(loader.dataset),
        "regression_loss": regression_loss_sum / len(loader.dataset),
        "terrain_accuracy": accuracy_score(terrain_true, terrain_pred) if len(terrain_true) else float("nan"),
        "terrain_macro_f1": f1_score(
            terrain_true, terrain_pred, labels=np.arange(len(VALID_TERRAINS)),
            average="macro", zero_division=0,
        ) if len(terrain_true) else float("nan"),
        "terrain_y_true": terrain_true,
        "terrain_y_pred": terrain_pred,
        "terrain_indices": np.asarray(terrain_indices, dtype=np.int64),
        "window_transition_accuracy": accuracy_score(labels, binary),
        "window_transition_f1": f1_score(labels, binary, zero_division=0),
        "transition_labels": labels,
        "transition_probabilities": probabilities,
        "predicted_offsets": offsets,
        "indices": np.asarray(all_indices, dtype=np.int64),
    }
    result["window_transition_roc_auc"] = (
        roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else float("nan")
    )
    result["window_transition_average_precision"] = (
        average_precision_score(labels, probabilities) if len(np.unique(labels)) == 2 else float("nan")
    )
    return result


def choose_validation_subject(test_subject: int, available: list[int]) -> int:
    ordered = [x for x in SUBJECT_IDS if x in available]
    return ordered[(ordered.index(test_subject) + 1) % len(ordered)]


def get_fold(manifest: pd.DataFrame, test_subject: int):
    available = sorted(int(x) for x in manifest["subject"].unique())
    val_subject = choose_validation_subject(test_subject, available)
    train_subjects = [x for x in available if x not in {test_subject, val_subject}]
    return (
        manifest[manifest["subject"].isin(train_subjects)].reset_index(drop=True),
        manifest[manifest["subject"] == val_subject].reset_index(drop=True),
        manifest[manifest["subject"] == test_subject].reset_index(drop=True),
        val_subject, train_subjects,
    )


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        var=scaler.var_.astype(np.float32),
        n_samples_seen=np.asarray(scaler.n_samples_seen_),
    )


def train_fold(
    manifest: pd.DataFrame, label_encoder: LabelEncoder, spec: BaselineSpec,
    test_subject: int, epochs: int, batch_size: int, num_workers: int,
):
    train_df, val_df, test_df, val_subject, train_subjects = get_fold(manifest, test_subject)
    fold_dir = spec.result_dir / f"test_subject_{test_subject:02d}"
    ckpt_dir = spec.checkpoint_dir / f"test_subject_{test_subject:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = fit_scaler(train_df)
    save_scaler(scaler, ckpt_dir / "imu_scaler.npz")

    train_loader = build_loader(MultiTaskDataset(train_df, scaler, label_encoder, spec, spec.uses_video), batch_size, num_workers, True)
    val_loader = build_loader(MultiTaskDataset(val_df, scaler, label_encoder, spec), batch_size, num_workers, False)
    test_loader = build_loader(MultiTaskDataset(test_df, scaler, label_encoder, spec), batch_size, num_workers, False)

    model = build_model(spec, len(label_encoder.classes_)).to(DEVICE)
    terrain_criterion = nn.CrossEntropyLoss(weight=class_weights(train_df, label_encoder))
    pos_weight = positive_weight(train_df)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    amp_scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    print("\n" + "=" * 90)
    print(f"{spec.name} | TEST S{test_subject:02d} | VAL S{val_subject:02d}")
    print("TRAIN: " + ", ".join(f"S{x:02d}" for x in train_subjects))
    print(f"Windows train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print("=" * 90)

    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_epoch, stale = -1, 0
    checkpoint_path = ckpt_dir / "best_multitask_model.pth"
    history = []

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, terrain_criterion, pos_weight, optimizer, amp_scaler)
        val_result = run_epoch(model, val_loader, terrain_criterion, pos_weight)

        val_windows = make_window_prediction_table(val_df, val_result)
        threshold, _ = choose_threshold(val_windows)
        val_recordings = aggregate_recording_predictions(val_windows, threshold)
        event = tolerance_metrics(val_recordings, VALIDATION_TOLERANCE_SEC)
        presence = evaluate_recordings(val_recordings)
        joint = 0.5 * val_result["terrain_macro_f1"] + 0.5 * event["f1"]
        scheduler.step(joint)

        score = (joint, val_result["terrain_macro_f1"], event["f1"], presence["presence_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_terrain_f1": train_result["terrain_macro_f1"],
            "validation_loss": val_result["loss"],
            "validation_terrain_f1": val_result["terrain_macro_f1"],
            "validation_presence_f1": presence["presence_f1"],
            "validation_event_f1": event["f1"],
            "validation_joint_score": joint,
            "threshold": threshold,
        })

        print(
            f"S{test_subject:02d} Epoch {epoch:03d}/{epochs} | "
            f"Train terrain F1 {train_result['terrain_macro_f1']:.4f} | "
            f"Val terrain F1 {val_result['terrain_macro_f1']:.4f} | "
            f"Val presence F1 {presence['presence_f1']:.4f} | "
            f"Val event F1 {event['f1']:.4f} | Joint {joint:.4f}"
        )

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_threshold": threshold,
                "label_classes": label_encoder.classes_.tolist(),
                "feature_cols": list(spec.feature_cols),
                "test_subject": test_subject,
                "validation_subject": val_subject,
                "train_subjects": train_subjects,
                "baseline": spec.name,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping after {PATIENCE} non-improving epochs.")
            break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    if best_epoch < 0:
        raise RuntimeError(f"No checkpoint saved for Subject {test_subject:02d}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_result = run_epoch(model, val_loader, terrain_criterion, pos_weight)
    test_result = run_epoch(model, test_loader, terrain_criterion, pos_weight)

    save_terrain_outputs(fold_dir, "validation", val_df, val_result, label_encoder)
    test_terrain = save_terrain_outputs(fold_dir, "test", test_df, test_result, label_encoder)

    val_windows = make_window_prediction_table(val_df, val_result)
    selected_threshold, threshold_table = choose_threshold(val_windows)
    threshold_table.to_csv(fold_dir / "validation_threshold_search.csv", index=False)

    test_windows = make_window_prediction_table(test_df, test_result)
    test_transition_metrics, test_recordings = save_transition_outputs(
        fold_dir, "test", test_windows, selected_threshold
    )

    metrics = {
        "test_subject": test_subject,
        "validation_subject": val_subject,
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "test_terrain_accuracy": test_result["terrain_accuracy"],
        "test_terrain_macro_f1": test_result["terrain_macro_f1"],
        "test_presence_accuracy": test_transition_metrics["presence_accuracy"],
        "test_presence_precision": test_transition_metrics["presence_precision"],
        "test_presence_recall": test_transition_metrics["presence_recall"],
        "test_presence_f1": test_transition_metrics["presence_f1"],
        "test_timing_mae_sec": test_transition_metrics["timing_mae_sec"],
        "test_timing_rmse_sec": test_transition_metrics["timing_rmse_sec"],
        "test_timing_median_absolute_error_sec": test_transition_metrics["timing_median_absolute_error_sec"],
    }
    for x in test_transition_metrics["tolerance_metrics"]:
        text = f"{x['tolerance_sec']:.2f}"
        metrics[f"test_event_precision_at_{text}s"] = x["precision"]
        metrics[f"test_event_recall_at_{text}s"] = x["recall"]
        metrics[f"test_event_f1_at_{text}s"] = x["f1"]

    pd.DataFrame([metrics]).to_csv(fold_dir / "fold_metrics.csv", index=False)
    return metrics, test_terrain, test_recordings


def finite_mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def save_summary(spec: BaselineSpec, fold_metrics, terrain_tables, transition_tables, label_encoder) -> None:
    spec.result_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(fold_metrics).sort_values("test_subject").reset_index(drop=True)
    metrics_df.to_csv(spec.result_dir / "loso_per_subject_results.csv", index=False)

    columns = [
        "test_terrain_accuracy", "test_terrain_macro_f1",
        "test_presence_accuracy", "test_presence_f1",
        "test_timing_mae_sec", "test_timing_rmse_sec",
        "test_event_f1_at_0.25s", "test_event_f1_at_0.50s", "test_event_f1_at_1.00s",
    ]
    summary_rows = []
    summary = {"baseline": spec.name, "number_of_completed_folds": len(metrics_df)}
    for column in columns:
        mean, std = finite_mean_std(metrics_df[column])
        summary_rows.append({"metric": column, "mean": mean, "sample_std": std})
        summary[f"{column}_mean"] = mean
        summary[f"{column}_sample_std"] = std
    pd.DataFrame(summary_rows).to_csv(spec.result_dir / "loso_summary_mean_std.csv", index=False)

    pooled_terrain = pd.concat(terrain_tables, ignore_index=True)
    pooled_terrain.to_csv(spec.result_dir / "loso_all_held_out_terrain_predictions.csv", index=False)
    label_ids = np.arange(len(label_encoder.classes_))
    summary["pooled_terrain_accuracy"] = float(accuracy_score(pooled_terrain["true_id"], pooled_terrain["pred_id"]))
    summary["pooled_terrain_macro_f1"] = float(f1_score(
        pooled_terrain["true_id"], pooled_terrain["pred_id"], labels=label_ids,
        average="macro", zero_division=0,
    ))

    pooled_transition = pd.concat(transition_tables, ignore_index=True)
    pooled_transition.to_csv(spec.result_dir / "loso_all_held_out_transition_recording_predictions.csv", index=False)
    pooled_metrics = evaluate_recordings(pooled_transition)
    summary["pooled_transition_recording_metrics"] = pooled_metrics

    with open(spec.result_dir / "loso_summary_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\nFINAL LOSO MULTI-TASK RESULTS")
    print("=============================")
    for column, label, suffix in [
        ("test_terrain_accuracy", "Terrain accuracy", ""),
        ("test_terrain_macro_f1", "Terrain macro-F1", ""),
        ("test_presence_accuracy", "Transition presence accuracy", ""),
        ("test_presence_f1", "Transition presence F1", ""),
        ("test_timing_mae_sec", "Transition timing MAE", " s"),
        ("test_timing_rmse_sec", "Transition timing RMSE", " s"),
    ]:
        mean, std = finite_mean_std(metrics_df[column])
        print(f"{label}: {mean:.4f} +/- {std:.4f}{suffix}")
    print(f"Pooled terrain accuracy: {summary['pooled_terrain_accuracy']:.4f}")
    print(f"Pooled terrain macro-F1: {summary['pooled_terrain_macro_f1']:.4f}")
    print(f"Pooled transition presence accuracy: {pooled_metrics['presence_accuracy']:.4f}")
    print(f"Pooled transition presence F1: {pooled_metrics['presence_f1']:.4f}")
    print(f"Results directory: {spec.result_dir}")


def train_loso(
    spec: BaselineSpec, only_test_subject: int | None,
    epochs: int, batch_size: int, num_workers: int,
) -> None:
    set_seed(RANDOM_SEED)
    manifest = load_manifest(spec)
    available = sorted(int(x) for x in manifest["subject"].unique())
    evaluation_subjects = [x for x in SUBJECT_IDS if x in available]

    if len(available) < 3:
        raise ValueError("At least three subjects are required for train/validation/test subject separation.")
    if only_test_subject is not None:
        if only_test_subject not in evaluation_subjects:
            raise ValueError(f"Subject {only_test_subject} is not available in the manifest.")
        evaluation_subjects = [only_test_subject]

    label_encoder = LabelEncoder().fit(VALID_TERRAINS)
    spec.label_path.write_text(json.dumps(label_encoder.classes_.tolist(), indent=2))

    print(f"Device: {DEVICE}")
    print(f"Baseline: {spec.name}")
    print(f"Evaluation subjects: {evaluation_subjects}")

    fold_metrics, terrain_tables, transition_tables = [], [], []
    for subject in evaluation_subjects:
        set_seed(RANDOM_SEED + subject)
        metrics, terrain_table, transition_table = train_fold(
            manifest, label_encoder, spec, subject, epochs, batch_size, num_workers
        )
        fold_metrics.append(metrics)
        terrain_tables.append(terrain_table)
        transition_tables.append(transition_table)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary(spec, fold_metrics, terrain_tables, transition_tables, label_encoder)
