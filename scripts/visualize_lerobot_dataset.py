#!/usr/bin/env python3
"""Visualize a converted LeRobot dataset using Rerun."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import rerun as rr


LOGGER = logging.getLogger(__name__)
DEFAULT_META_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset episode")
    parser.add_argument("--data-dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--action-labels", nargs="*", help="Optional per-dimension labels for the action vector")
    parser.add_argument(
        "--no-block",
        dest="block",
        action="store_false",
        help="Return immediately after streaming frames (default keeps process alive)",
    )
    parser.set_defaults(block=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(meta_dir: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    task_file = meta_dir / "tasks.jsonl"
    if not task_file.exists():
        return tasks
    with task_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            index = int(payload["task_index"])
            tasks[index] = payload.get("task", "")
    return tasks


def resolve_episode_paths(info: dict[str, Any], root: Path, episode_index: int) -> tuple[Path, int]:
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    data_template = info.get("data_path", DEFAULT_DATA_TEMPLATE)
    rel_data = data_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
    data_path = root / rel_data
    return data_path, episode_chunk


DEFAULT_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}.mp4"


def gather_image_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return sorted([name for name, spec in features.items() if spec.get("dtype") == "image"])


def gather_numeric_keys(info: dict[str, Any], columns: Iterable[str]) -> list[str]:
    features = info.get("features", {})
    numeric: list[str] = []
    for name in columns:
        if name not in features:
            continue
        spec = features[name]
        if spec.get("dtype") in {"image", "video"}:
            continue
        if name in DEFAULT_META_FEATURES:
            continue
        numeric.append(name)
    return sorted(numeric)


def open_video_captures(
    info: dict[str, Any],
    dataset_dir: Path,
    image_keys: Iterable[str],
    episode_chunk: int,
    episode_index: int,
) -> dict[str, Any]:
    template = info.get("video_path") or DEFAULT_VIDEO_TEMPLATE
    captures: dict[str, Any] = {}
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        LOGGER.warning("opencv-python is required for video playback; skipping cameras")
        return captures
    for image_key in image_keys:
        rel_video = template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            image_key=image_key,
        )
        video_path = dataset_dir / rel_video
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            LOGGER.warning("Failed to open video %s", video_path)
            cap.release()
            continue
        captures[image_key] = cap
    return captures


def log_numeric_state(feature_name: str, value: Any, labels: list[str] | None = None) -> None:
    arr = np.asarray(value)
    if arr.size == 0:
        return
    flat = arr.reshape(-1)
    for idx, val in enumerate(flat):
        try:
            scalar = float(val)
        except (TypeError, ValueError):
            continue
        label = None
        if labels and idx < len(labels):
            label = labels[idx]
        rr.log(
            f"states/{feature_name}_{idx}",
            rr.TimeSeriesScalar(scalar, label=label),
        )


def log_camera_frames(captures: dict[str, Any]) -> None:
    if not captures:
        return
    import cv2  # type: ignore

    for key, cap in captures.items():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log(f"camera/{key}", rr.Image(frame_rgb))


def visualize_episode(dataset_dir: Path, episode_index: int, action_labels: list[str] | None) -> None:
    meta_dir = dataset_dir / "meta"
    info = load_json(meta_dir / "info.json")
    tasks = load_tasks(meta_dir)
    data_path, episode_chunk = resolve_episode_paths(info, dataset_dir, episode_index)
    if not data_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {data_path}")
    df = pd.read_parquet(data_path)
    run_name = f"{dataset_dir.name}/episode_{episode_index:06d}"
    rr.init(run_name, spawn=True)
    image_keys = gather_image_keys(info)
    captures = open_video_captures(info, dataset_dir, image_keys, episode_chunk, episode_index)
    numeric_keys = gather_numeric_keys(info, df.columns)

    for frame_idx, row in df.iterrows():
        rr.set_time_sequence("frame", int(frame_idx))
        if "timestamp" in row:
            rr.set_time_seconds("timestamp", float(row["timestamp"]))
        log_camera_frames(captures)
        for key in numeric_keys:
            labels = action_labels if key == "action" else None
            log_numeric_state(key, row[key], labels)
        if "task_index" in row:
            task_name = tasks.get(int(row["task_index"]), None)
            if task_name:
                rr.log("states/task", rr.TextLog(task_name))

    for cap in captures.values():
        cap.release()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    action_labels = args.action_labels if args.action_labels else None
    visualize_episode(Path(args.data_dir).expanduser(), args.episode, action_labels)
    if args.block:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
