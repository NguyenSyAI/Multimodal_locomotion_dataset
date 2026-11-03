from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset

from config import (
    DATA_ROOT, SUBJECT_IDS, VALID_TERRAINS, TERRAIN_ALIASES,
    IMU_WINDOW_ROWS, IMU_STEP_ROWS, TRANSITION_SKIP_SEC,
    VIDEO_NUM_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, RANDOM_SEED,
    BaselineSpec,
)


def canonical_terrain(value: str) -> str | None:
    text = str(value).strip()
    while text.lower().startswith("og_"):
        text = text[3:]
    direct = {x.lower(): x for x in VALID_TERRAINS}
    return direct.get(text.lower(), TERRAIN_ALIASES.get(text.lower()))


def first_recognized_terrain(descriptor: str) -> str | None:
    text = str(descriptor).strip()
    while text.lower().startswith("og_"):
        text = text[3:]

    direct = canonical_terrain(text)
    if direct is not None:
        return direct

    tokens = text.split("_")
    for token in tokens:
        terrain = canonical_terrain(token)
        if terrain is not None:
            return terrain

    for i in range(len(tokens) - 1):
        terrain = canonical_terrain(f"{tokens[i]}_{tokens[i + 1]}")
        if terrain is not None:
            return terrain
    return None


def parse_recording_stem(value: str | Path) -> dict[str, Any]:
    filename = Path(value).name
    stem = re.sub(
        r"^(IMU|ShankCam|ChestCam)_", "", Path(filename).stem,
        flags=re.IGNORECASE,
    )
    match = re.match(
        r"^S0*(?P<subject>\d+)_(?P<terrain_part>.+)_"
        r"(?P<speed>Fast|Slow|Normal)_T(?P<trial>\d+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot parse recording filename: {filename}")

    terrain_part = match.group("terrain_part")
    parts = re.split(r"_to_", terrain_part, maxsplit=1, flags=re.IGNORECASE)
    is_transition = len(parts) == 2

    before = first_recognized_terrain(parts[0])
    after = first_recognized_terrain(parts[1]) if is_transition else None
    if before is None or (is_transition and after is None):
        raise ValueError(f"Could not identify terrain(s) in {filename}")

    subject = int(match.group("subject"))
    speed = match.group("speed").capitalize()
    trial = int(match.group("trial"))
    terrain_key = f"{before.lower()}_to_{after.lower()}" if is_transition else before.lower()

    return {
        "subject": subject,
        "terrain_before": before,
        "terrain_after": after,
        "is_transition": is_transition,
        "speed": speed,
        "trial": trial,
        "canonical_key": f"s{subject}_{terrain_key}_{speed.lower()}_t{trial}",
        "file_stem": Path(filename).stem,
        "filename": filename,
    }


def try_recording_key(value: object) -> str | None:
    try:
        return parse_recording_stem(str(value))["canonical_key"]
    except Exception:
        return None


def discover_subject_config(subject: int, spec: BaselineSpec) -> dict[str, Path] | None:
    subject_dir = next(
        (p for p in [DATA_ROOT / f"Subject {subject}", DATA_ROOT / f"Subject {subject:02d}"] if p.is_dir()),
        None,
    )
    if subject_dir is None:
        print(f"[WARNING] Missing subject directory for Subject {subject:02d}")
        return None

    imu_dir = next(
        (p for p in [subject_dir / f"S{subject}_Synced", subject_dir / f"S{subject:02d}_Synced"] if p.is_dir()),
        None,
    )
    if imu_dir is None:
        matches = sorted(p for p in subject_dir.glob("*Synced*") if p.is_dir())
        imu_dir = matches[0] if matches else None
    if imu_dir is None:
        print(f"[WARNING] Missing IMU directory for Subject {subject:02d}")
        return None

    result = {"subject_dir": subject_dir, "imu_dir": imu_dir}
    if spec.uses_video:
        kind = spec.video_kind
        video_dir = next(
            (p for p in [subject_dir / f"S{subject}_{kind}", subject_dir / f"S{subject:02d}_{kind}"] if p.is_dir()),
            None,
        )
        if video_dir is None:
            matches = sorted(p for p in subject_dir.glob(f"*{kind}*") if p.is_dir())
            video_dir = matches[0] if matches else None
        if video_dir is None:
            print(f"[WARNING] Missing {kind} directory for Subject {subject:02d}")
            return None
        result["video_dir"] = video_dir
    return result


def build_video_index(video_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))):
        key = try_recording_key(path.name)
        if key is not None:
            index.setdefault(key, []).append(path)
    return index


def get_matching_video_path(
    csv_path: Path,
    video_dir: Path,
    video_index: dict[str, list[Path]],
    video_kind: str,
) -> Path | None:
    exact_name = re.sub(
        r"\.csv$", ".mp4",
        csv_path.name.replace("IMU_", f"{video_kind}_", 1),
        flags=re.IGNORECASE,
    )
    exact = video_dir / exact_name
    if exact.is_file():
        return exact

    for candidate in list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4")):
        if candidate.name.lower() == exact_name.lower():
            return candidate

    key = try_recording_key(csv_path.name)
    candidates = video_index.get(key, []) if key is not None else []
    if len(candidates) > 1:
        print(f"[WARNING] Multiple videos matched {csv_path.name}; using {candidates[0].name}")
    return candidates[0] if candidates else None


def find_signal_index_path(subject_dir: Path) -> Path | None:
    patterns = [
        "*Signal_Index*.csv", "*signal_index*.csv", "*Signal*Index*.csv",
        "*Signal_Index*.xlsx", "*signal_index*.xlsx", "*Signal*Index*.xlsx",
        "*Signal_Index*.xls", "*signal_index*.xls", "*Signal*Index*.xls",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(subject_dir.glob(pattern))
    matches = sorted(set(matches))
    return matches[0] if matches else None


def read_signal_index(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported Signal_Index format: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_transition_time(csv_path: Path, signal_index_path: Path | None) -> float | None:
    if signal_index_path is None:
        return None
    try:
        index_df = read_signal_index(signal_index_path)
    except Exception as error:
        print(f"[WARNING] Could not read {signal_index_path}: {error}")
        return None

    transition_col = next(
        (str(c) for c in index_df.columns if str(c).strip().lower() == "transition_1"),
        None,
    )
    if transition_col is None:
        return None

    target_key = try_recording_key(csv_path.name)
    if target_key is None:
        return None

    matched = []
    for row_index, row in index_df.iterrows():
        row_keys = {
            key
            for value in row.values
            if not pd.isna(value)
            for key in [try_recording_key(value)]
            if key is not None
        }
        if target_key in row_keys:
            matched.append(int(row_index))

    if not matched:
        return None
    if len(matched) > 1:
        print(f"[WARNING] Multiple Signal_Index rows matched {csv_path.name}; using first")

    value = index_df.loc[matched[0], transition_col]
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.upper() in {"", "NA", "N/A", "NONE", "NAN"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def validate_imu_dataframe(df: pd.DataFrame, csv_path: Path, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    required = list(feature_cols) + ["imu_sec", "video_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    original = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < IMU_WINDOW_ROWS:
        raise ValueError(f"Fewer than {IMU_WINDOW_ROWS} valid rows")
    if original != len(df):
        print(f"[INFO] Removed {original - len(df)} invalid rows from {csv_path.name}")
    return df


def terrain_target_for_window(
    start_time: float,
    end_time: float,
    terrain_before: str,
    terrain_after: str | None,
    is_transition: bool,
    transition_time: float | None,
) -> tuple[str | None, int]:
    if not is_transition:
        return terrain_before, 1
    if transition_time is None or terrain_after is None:
        return None, 0
    if end_time < transition_time - TRANSITION_SKIP_SEC:
        return terrain_before, 1
    if start_time > transition_time + TRANSITION_SKIP_SEC:
        return terrain_after, 1
    return None, 0


def choose_frame_rows() -> np.ndarray:
    return np.round(np.linspace(0, IMU_WINDOW_ROWS - 1, VIDEO_NUM_FRAMES)).astype(np.int64)


def build_recording_windows(
    dataframe: pd.DataFrame,
    csv_path: Path,
    subject: int,
    info: dict[str, Any],
    transition_time: float | None,
    spec: BaselineSpec,
    video_path: Path | None = None,
) -> list[dict[str, Any]]:
    frame_rows = choose_frame_rows() if spec.uses_video else None
    has_transition = transition_time is not None
    samples = []

    for start_row in range(0, len(dataframe) - IMU_WINDOW_ROWS + 1, IMU_STEP_ROWS):
        end_row = start_row + IMU_WINDOW_ROWS
        window = dataframe.iloc[start_row:end_row]
        start_time = float(window["video_sec"].iloc[0])
        end_time = float(window["video_sec"].iloc[-1])
        if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
            continue

        transition_label = 0.0
        transition_offset = -1.0
        if has_transition and start_time <= float(transition_time) <= end_time:
            transition_label = 1.0
            transition_offset = float(np.clip(
                (float(transition_time) - start_time) / (end_time - start_time), 0.0, 1.0
            ))

        terrain, terrain_valid = terrain_target_for_window(
            start_time, end_time,
            info["terrain_before"], info["terrain_after"],
            info["is_transition"], transition_time,
        )

        sample = {
            "sample_id": f"S{subject:02d}_{info['canonical_key']}_{start_row:07d}_{end_row:07d}",
            "subject": subject,
            "recording_key": info["canonical_key"],
            "recording_stem": info["file_stem"],
            "csv_path": str(csv_path),
            "imu": window[list(spec.feature_cols)].to_numpy(dtype=np.float32),
            "window_start_time": start_time,
            "window_end_time": end_time,
            "terrain": terrain or "",
            "terrain_valid": int(terrain_valid),
            "recording_has_transition": int(has_transition),
            "true_transition_time": float(transition_time) if has_transition else np.nan,
            "transition_label": transition_label,
            "transition_offset": transition_offset,
        }

        if spec.uses_video:
            selected = window.iloc[frame_rows]
            timestamps = selected["video_sec"].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0):
                continue
            sample["video_path"] = str(video_path)
            sample["video_timestamps"] = timestamps

        samples.append(sample)
    return samples


def build_subject_samples(subject: int, spec: BaselineSpec) -> list[dict[str, Any]]:
    config = discover_subject_config(subject, spec)
    if config is None:
        return []

    signal_index = find_signal_index_path(config["subject_dir"])
    video_index = build_video_index(config["video_dir"]) if spec.uses_video else None
    csv_paths = sorted(list(config["imu_dir"].glob("IMU_*.csv")) + list(config["imu_dir"].glob("IMU_*.CSV")))
    print(f"\nSubject {subject:02d}: {len(csv_paths)} CSV files")

    all_samples = []
    for csv_path in csv_paths:
        try:
            info = parse_recording_stem(csv_path.name)
        except ValueError as error:
            print(f"[SKIP] {error}")
            continue

        video_path = None
        if spec.uses_video:
            video_path = get_matching_video_path(csv_path, config["video_dir"], video_index, spec.video_kind)
            if video_path is None:
                print(f"[SKIP] No matching {spec.video_kind} video for {csv_path.name}")
                continue

        transition_time = find_transition_time(csv_path, signal_index)
        if info["is_transition"] and transition_time is None:
            print(f"[SKIP] Transition recording has no usable Transition_1: {csv_path.name}")
            continue
        if not info["is_transition"]:
            transition_time = None

        try:
            df = validate_imu_dataframe(pd.read_csv(csv_path), csv_path, spec.feature_cols)
        except Exception as error:
            print(f"[SKIP] Could not use {csv_path}: {error}")
            continue

        samples = build_recording_windows(df, csv_path, subject, info, transition_time, spec, video_path)
        all_samples.extend(samples)
        print(f"[INFO] {csv_path.name} | transition={transition_time} | windows={len(samples)}")

    print(f"  Total window descriptions: {len(all_samples)}")
    return all_samples


def read_frames(video_path: str | Path, timestamps: np.ndarray) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    frame_indices = np.clip(np.rint(timestamps * fps).astype(np.int64), 0, total_frames - 1)
    frames, cache = [], {}
    for index in frame_indices:
        index = int(index)
        if index not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Could not read frame {index} from {video_path}")
            cache[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cache[index].copy())
    cap.release()
    return frames


def resize_video_frames(frames: list[np.ndarray]) -> np.ndarray:
    processed = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = max(VIDEO_HEIGHT / h, VIDEO_WIDTH / w)
        new_h = max(VIDEO_HEIGHT, int(round(h * scale)))
        new_w = max(VIDEO_WIDTH, int(round(w * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (new_h - VIDEO_HEIGHT) // 2
        left = (new_w - VIDEO_WIDTH) // 2
        crop = resized[top:top + VIDEO_HEIGHT, left:left + VIDEO_WIDTH]
        if crop.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
            crop = cv2.resize(crop, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LINEAR)
        processed.append(np.ascontiguousarray(crop, dtype=np.uint8))
    return np.stack(processed, axis=0)


def prepare_cache(spec: BaselineSpec, overwrite: bool = False) -> None:
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    spec.sample_dir.mkdir(parents=True, exist_ok=True)

    if spec.manifest_path.is_file() and not overwrite:
        print(f"Manifest already exists: {spec.manifest_path}")
        print("Use --overwrite-cache to rebuild it.")
        return

    samples = []
    for subject in SUBJECT_IDS:
        samples.extend(build_subject_samples(subject, spec))
    if not samples:
        raise RuntimeError("No multitask samples were constructed.")

    manifest_rows = []
    for number, sample in enumerate(samples, start=1):
        subject_dir = spec.sample_dir / f"subject_{sample['subject']:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        npz_path = subject_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", sample["sample_id"]) + ".npz")

        recreate = overwrite
        if npz_path.is_file() and not recreate:
            try:
                cached = np.load(npz_path, allow_pickle=False)
                valid = cached["imu"].shape == (IMU_WINDOW_ROWS, len(spec.feature_cols))
                if spec.uses_video:
                    valid = valid and cached["frames"].shape == (
                        VIDEO_NUM_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3
                    )
                recreate = not valid
            except Exception:
                recreate = True
        if recreate and npz_path.exists():
            npz_path.unlink()

        if not npz_path.exists():
            try:
                arrays = {"imu": sample["imu"].astype(np.float32)}
                if spec.uses_video:
                    arrays["frames"] = resize_video_frames(
                        read_frames(sample["video_path"], sample["video_timestamps"])
                    )
                np.savez_compressed(npz_path, **arrays)
            except Exception as error:
                print(f"[SKIP] {sample['sample_id']}: {error}")
                continue

        row = {
            key: sample[key]
            for key in [
                "sample_id", "subject", "recording_key", "recording_stem",
                "window_start_time", "window_end_time", "terrain", "terrain_valid",
                "recording_has_transition", "true_transition_time",
                "transition_label", "transition_offset",
            ]
        }
        row["npz_relpath"] = npz_path.relative_to(spec.cache_dir).as_posix()
        manifest_rows.append(row)

        if number % 250 == 0 or number == len(samples):
            print(f"Cached {number}/{len(samples)} windows")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(spec.manifest_path, index=False)
    spec.label_path.write_text(json.dumps(VALID_TERRAINS, indent=2))

    print(f"\nCache prepared: {spec.manifest_path}")
    print(f"Windows: {len(manifest)}")
    print("\nWindows by subject:")
    print(manifest["subject"].value_counts().sort_index())


def load_manifest(spec: BaselineSpec) -> pd.DataFrame:
    if not spec.manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {spec.manifest_path}\nRun --mode prepare first.")

    manifest = pd.read_csv(spec.manifest_path, keep_default_na=False)
    required = {
        "sample_id", "subject", "recording_key", "recording_stem", "npz_relpath",
        "window_start_time", "window_end_time", "terrain", "terrain_valid",
        "recording_has_transition", "true_transition_time", "transition_label", "transition_offset",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}. Rebuild the cache.")

    manifest["subject"] = manifest["subject"].astype(int)
    manifest["terrain_valid"] = manifest["terrain_valid"].astype(int)
    for column in [
        "window_start_time", "window_end_time", "recording_has_transition",
        "true_transition_time", "transition_label", "transition_offset",
    ]:
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")

    manifest["npz_path"] = manifest["npz_relpath"].map(lambda x: str(spec.cache_dir / x))
    missing_files = [x for x in manifest["npz_path"] if not Path(x).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} cache files are missing. Re-run prepare --overwrite-cache.")

    unknown = set(manifest.loc[manifest["terrain_valid"] == 1, "terrain"].astype(str)) - set(VALID_TERRAINS)
    if unknown:
        raise ValueError(f"Unsupported terrain labels in manifest: {sorted(unknown)}")
    return manifest


def fit_scaler(train_manifest: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    for path in train_manifest["npz_path"]:
        scaler.partial_fit(np.load(path, allow_pickle=False)["imu"].astype(np.float32))
    return scaler


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        scaler: StandardScaler,
        label_encoder: LabelEncoder,
        spec: BaselineSpec,
        augment_video: bool = False,
    ):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.spec = spec
        self.augment_video = augment_video
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

        self.terrain_ids = np.full(len(self.manifest), -1, dtype=np.int64)
        valid = self.manifest["terrain_valid"].to_numpy(dtype=np.int64) == 1
        if np.any(valid):
            self.terrain_ids[valid] = label_encoder.transform(
                self.manifest.loc[valid, "terrain"].astype(str)
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        data = np.load(row["npz_path"], allow_pickle=False)

        imu = ((data["imu"].astype(np.float32) - self.mean) / self.scale).astype(np.float32)

        if self.spec.uses_video:
            frames = data["frames"].astype(np.float32) / 255.0
            if self.augment_video and random.random() < 0.5:
                frames = frames[:, :, ::-1, :].copy()
            frames = np.transpose(frames, (0, 3, 1, 2))
            frames = np.ascontiguousarray((frames - 0.5) / 0.5, dtype=np.float32)
            frames_tensor = torch.from_numpy(frames)
        else:
            frames_tensor = torch.empty(0, dtype=torch.float32)

        return (
            torch.from_numpy(imu),
            frames_tensor,
            torch.tensor(int(self.terrain_ids[index]), dtype=torch.long),
            torch.tensor(bool(row["terrain_valid"]), dtype=torch.bool),
            torch.tensor(float(row["transition_label"]), dtype=torch.float32),
            torch.tensor(float(row["transition_offset"]), dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )
