#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer as DiffusionReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Features


CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
INFO_PATH = "meta/info.json"
TASKS_PATH = "meta/tasks.jsonl"
EPISODES_PATH = "meta/episodes.jsonl"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
STATS_PATH = "meta/stats.json"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}.mp4"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

_DATASETS_CACHE: dict[str, Any] | None = None


class VideoExportError(RuntimeError):
    pass


def _build_local_datasets_fallback() -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    class SimpleFeatures(dict):
        """Lightweight stand-in for datasets.Features."""

    class SimpleImage:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class SimpleValue:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

    class SimpleSequence:
        def __init__(self, length: int, feature: Any) -> None:
            self.length = length
            self.feature = feature

    class SimpleArray:
        def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
            self.shape = shape
            self.dtype = dtype

    class SimpleDataset:
        def __init__(self, data: dict[str, list[Any]]) -> None:
            self.data = data

        @staticmethod
        def _normalize_column(values: list[Any]) -> list[Any]:
            normalized: list[Any] = []
            for value in values:
                if isinstance(value, np.ndarray):
                    normalized.append(value.tolist())
                else:
                    normalized.append(value)
            return normalized

        @classmethod
        def from_dict(cls, data_dict: dict[str, list[Any]], features: Any | None = None) -> "SimpleDataset":
            normalized = {key: cls._normalize_column(values) for key, values in data_dict.items()}
            return cls(normalized)

        def map(self, fn, batched: bool = False):  # type: ignore[override]
            if callable(fn):
                self.data = fn(self.data)
            return self

        def to_parquet(self, path: str) -> None:
            arrays = {key: pa.array(values) for key, values in self.data.items()}
            table = pa.table(arrays)
            pq.write_table(table, path)

    def embed_table_storage(batch: dict[str, list[Any]]):
        return batch

    return {
        "Dataset": SimpleDataset,
        "Features": SimpleFeatures,
        "Image": SimpleImage,
        "Sequence": SimpleSequence,
        "Value": SimpleValue,
        "Array2D": SimpleArray,
        "Array3D": SimpleArray,
        "Array4D": SimpleArray,
        "Array5D": SimpleArray,
        "embed_table_storage": embed_table_storage,
    }


def require_datasets_objects() -> dict[str, Any]:
    global _DATASETS_CACHE
    if _DATASETS_CACHE is None:
        try:
            from datasets import Array2D, Array3D, Array4D, Array5D, Dataset, Features, Image, Sequence, Value
            from datasets.table import embed_table_storage

            _DATASETS_CACHE = {
                "Dataset": Dataset,
                "Features": Features,
                "Image": Image,
                "Sequence": Sequence,
                "Value": Value,
                "Array2D": Array2D,
                "Array3D": Array3D,
                "Array4D": Array4D,
                "Array5D": Array5D,
                "embed_table_storage": embed_table_storage,
            }
        except ModuleNotFoundError:
            logging.warning("'datasets' package not found; using a minimal local fallback implementation.")
            _DATASETS_CACHE = _build_local_datasets_fallback()
    return _DATASETS_CACHE


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonline(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def load_image_as_numpy(path: str, dtype: np.dtype = np.uint8) -> np.ndarray:
    with PIL.Image.open(path) as img:
        arr = np.array(img, dtype=dtype)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return arr


def estimate_num_samples(dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    if data_len == 0:
        return []
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))
    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        img = load_image_as_numpy(path, dtype=np.uint8)
        img = auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
        images[i] = img
    return images if images is not None else np.empty((0,))


def get_feature_stats(array: np.ndarray, axis: tuple | int, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([array.shape[0]]),
    }


def compute_episode_stats(episode_data: dict[str, Any], features: dict) -> dict:
    stats = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        if ft["dtype"] == "image":
            img_array = sample_images(episode_data[key])
            if img_array.size == 0:
                continue
            axes = (0, 2, 3)
            keepdims = True
            values = img_array
        else:
            values = episode_data[key]
            axes = 0
            keepdims = values.ndim == 1
        stats[key] = get_feature_stats(values, axis=axes, keepdims=keepdims)
        if ft["dtype"] == "image":
            stats[key] = {k: (v if k == "count" else np.squeeze(v / 255.0, axis=0)) for k, v in stats[key].items()}
    return stats


def aggregate_feature_stats(stats_ft_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count
    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
def merge_stats(existing: dict[str, dict] | None, new_stats: dict[str, dict]) -> dict[str, dict]:
    if existing is None:
        return {key: {k: v.copy() for k, v in stats.items()} for key, stats in new_stats.items()}
    keys = set(existing) | set(new_stats)
    aggregated: dict[str, dict] = {}
    for key in keys:
        subset: list[dict[str, np.ndarray]] = []
        if key in existing:
            subset.append(existing[key])
        if key in new_stats:
            subset.append(new_stats[key])
        aggregated[key] = aggregate_feature_stats(subset)
    return aggregated


def serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    serialized = {}
    for key, stat_dict in stats.items():
        serialized[key] = {name: value.tolist() for name, value in stat_dict.items()}
    return serialized


def create_empty_dataset_info(codebase_version: str, fps: int, robot_type: str | None, features: dict) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": None,
        "features": features,
    }


def get_hf_features_from_features(features: dict) -> Features:
    objs = require_datasets_objects()
    FeaturesCls = objs["Features"]
    ImageCls = objs["Image"]
    SequenceCls = objs["Sequence"]
    ValueCls = objs["Value"]
    Array2DCls = objs["Array2D"]
    Array3DCls = objs["Array3D"]
    Array4DCls = objs["Array4D"]
    Array5DCls = objs["Array5D"]
    hf_features = {}
    for key, ft in features.items():
        dtype = ft["dtype"]
        shape = ft["shape"]
        if dtype == "video":
            continue
        if dtype == "image":
            hf_features[key] = ImageCls()
            continue
        if shape == (1,):
            hf_features[key] = ValueCls(dtype=dtype)
        elif len(shape) == 1:
            hf_features[key] = SequenceCls(length=shape[0], feature=ValueCls(dtype=dtype))
        elif len(shape) == 2:
            hf_features[key] = Array2DCls(shape=shape, dtype=dtype)
        elif len(shape) == 3:
            hf_features[key] = Array3DCls(shape=shape, dtype=dtype)
        elif len(shape) == 4:
            hf_features[key] = Array4DCls(shape=shape, dtype=dtype)
        elif len(shape) == 5:
            hf_features[key] = Array5DCls(shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported shape {shape} for feature {key}")
    return FeaturesCls(hf_features)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features
    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"
    return error_message


def validate_feature_numpy_array(name: str, expected_dtype: str, expected_shape: tuple[int, ...], value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray):
        return f"Feature '{name}' is not a numpy array."
    if expected_dtype != "image" and value.dtype != np.dtype(expected_dtype):
        return f"Feature '{name}' expected dtype {expected_dtype} but got {value.dtype}"
    if expected_shape != value.shape:
        return f"Feature '{name}' expected shape {expected_shape} but got {value.shape}"
    return ""


def validate_feature_image_or_video(name: str, expected_shape: tuple[int, ...], value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return ""
    if not isinstance(value, np.ndarray):
        return f"Feature '{name}' should be an array or path."
    if value.shape != expected_shape:
        return f"Image '{name}' expected shape {expected_shape} but got {value.shape}"
    return ""


def validate_feature_dtype_and_shape(name: str, feature: dict, value: Any) -> str:
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    return validate_feature_numpy_array(name, expected_dtype, expected_shape, np.asarray(value))


def validate_frame(frame: dict, features: dict) -> None:
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)
    if "task" not in actual_features:
        raise ValueError("Frame must include 'task'")
    actual_for_validation = actual_features - {"task"}
    error_message = validate_features_presence(actual_for_validation, expected_features)
    common_features = actual_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])
    if error_message:
        raise ValueError(error_message)


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    if "size" not in episode_buffer or "task" not in episode_buffer:
        raise ValueError("Episode buffer missing required keys")
    if episode_buffer["episode_index"] != total_episodes:
        raise NotImplementedError("Episode indices must be sequential")
    if episode_buffer["size"] == 0:
        raise ValueError("Episode buffer is empty")
    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if buffer_keys != set(features):
        raise ValueError("Episode buffer features do not match dataset features")


def write_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    img = PIL.Image.fromarray(array)
    img.save(path)


def write_video_from_images(image_paths: list[Path], path: Path, fps: int) -> None:
    if not image_paths:
        raise VideoExportError(f"No frames provided for video export at {path}")
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise VideoExportError("OpenCV (cv2) is required for video export") from exc
    frame_shape: tuple[int, int] | None = None
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for img_path in image_paths:
            frame = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame_shape is None:
                height, width = frame.shape[:2]
                frame_shape = (width, height)
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), fourcc, fps, frame_shape)
                if not writer.isOpened():
                    raise VideoExportError(f"Unable to open video writer for {path}")
            writer.write(frame)
        if writer is None:
            raise VideoExportError(f"No readable frames found for {path}")
    finally:
        if writer is not None:
            writer.release()


class SimpleLeRobotDataset:
    def __init__(
        self,
        repo_id: str,
        root: Path,
        fps: int,
        features: dict,
        robot_type: str | None = None,
    ) -> None:
        if root.exists():
            raise FileExistsError(f"Output directory {root} already exists")
        self.repo_id = repo_id
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        merged_features = {**features, **DEFAULT_FEATURES}
        self.features = merged_features
        self.video_keys = sorted(key for key, spec in merged_features.items() if spec.get("dtype") == "image")
        self._video_export_disabled = False
        self.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, merged_features)
        if self.video_keys:
            self.info["video_path"] = DEFAULT_VIDEO_PATH
        write_json(self.info, self.root / INFO_PATH)
        self.tasks: dict[int, str] = {}
        self.task_to_index: dict[str, int] = {}
        self.stats: dict[str, dict] | None = None
        self.total_frames = 0
        self.fps = fps
        self.hf_features = get_hf_features_from_features(self.features)
        self.episode_buffer = self.create_episode_buffer()
        for rel in (TASKS_PATH, EPISODES_PATH, EPISODES_STATS_PATH):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    def create_episode_buffer(self) -> dict:
        ep_idx = self.info["total_episodes"]
        buffer = {key: [] for key in self.features}
        buffer["size"] = 0
        buffer["task"] = []
        buffer["episode_index"] = ep_idx
        return buffer

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.info["chunks_size"]

    def _get_data_path(self, episode_index: int) -> Path:
        chunk = self._episode_chunk(episode_index)
        rel = DEFAULT_PARQUET_PATH.format(episode_chunk=chunk, episode_index=episode_index)
        return self.root / rel

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        rel = DEFAULT_IMAGE_PATH.format(image_key=image_key, episode_index=episode_index, frame_index=frame_index)
        return self.root / rel

    def _get_video_file_path(self, episode_index: int, image_key: str) -> Path:
        chunk = self._episode_chunk(episode_index)
        rel = DEFAULT_VIDEO_PATH.format(episode_chunk=chunk, image_key=image_key, episode_index=episode_index)
        return self.root / rel

    def _add_task(self, task: str) -> int:
        if task in self.task_to_index:
            return self.task_to_index[task]
        task_index = len(self.tasks)
        self.tasks[task_index] = task
        self.task_to_index[task] = task_index
        append_jsonline(self.root / TASKS_PATH, {"task_index": task_index, "task": task})
        return task_index

    def add_frame(self, frame: dict) -> None:
        validate_frame(frame, self.features)
        for key, value in list(frame.items()):
            if isinstance(value, np.ndarray):
                frame[key] = value
        buffer = self.episode_buffer
        frame_index = buffer["size"]
        timestamp = frame.get("timestamp")
        if timestamp is None:
            timestamp = frame_index / self.fps
        buffer["timestamp"].append(float(timestamp))
        buffer["frame_index"].append(int(frame_index))
        buffer["task"].append(frame.pop("task"))
        for key in self.features:
            if key in DEFAULT_FEATURES:
                continue
            value = frame.get(key)
            if value is None:
                raise ValueError(f"Missing feature {key} in frame")
            spec = self.features[key]
            if spec["dtype"] == "video":
                raise NotImplementedError("Video features are not supported by this converter")
            if spec["dtype"] == "image":
                img_path = self._get_image_file_path(self.episode_buffer["episode_index"], key, frame_index)
                write_image(value, img_path)
                buffer[key].append(str(img_path))
            else:
                buffer[key].append(value)
        buffer["size"] += 1

    def _cleanup_episode_images(self, episode_index: int) -> None:
        img_root = self.root / "images"
        if img_root.is_dir():
            shutil.rmtree(img_root)

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        objs = require_datasets_objects()
        DatasetCls = objs["Dataset"]
        embed_fn = objs["embed_table_storage"]
        data_dict = {key: episode_buffer[key] for key in self.hf_features}
        dataset = DatasetCls.from_dict(data_dict, features=self.hf_features)
        dataset = dataset.map(embed_fn, batched=False)
        data_path = self._get_data_path(episode_index)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(data_path))

    def save_episode(self) -> None:
        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.info["total_episodes"], self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(self.total_frames, self.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        for task in set(tasks):
            self._add_task(task)
        episode_buffer["task_index"] = np.array([self.task_to_index[t] for t in tasks])
        episode_buffer["timestamp"] = np.asarray(episode_buffer["timestamp"], dtype=np.float32)
        episode_buffer["frame_index"] = np.asarray(episode_buffer["frame_index"], dtype=np.int64)
        episode_buffer["index"] = np.asarray(episode_buffer["index"], dtype=np.int64)
        episode_buffer["task_index"] = np.asarray(episode_buffer["task_index"], dtype=np.int64)
        episode_buffer["episode_index"] = np.asarray(episode_buffer["episode_index"], dtype=np.int64)
        for key, spec in self.features.items():
            if key in DEFAULT_FEATURES or spec["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        self._save_episode_table(episode_buffer, episode_index)
        created_videos = self._export_episode_videos(episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self.stats = merge_stats(self.stats, ep_stats)
        append_jsonline(
            self.root / EPISODES_PATH,
            {"episode_index": episode_index, "tasks": list(set(tasks)), "length": episode_length},
        )
        append_jsonline(
            self.root / EPISODES_STATS_PATH,
            {"episode_index": episode_index, "stats": serialize_stats(ep_stats)},
        )
        self.total_frames += episode_length
        self.info["total_episodes"] += 1
        self.info["total_frames"] = self.total_frames
        self.info["total_tasks"] = len(self.tasks)
        self.info["total_videos"] += created_videos
        if created_videos > 0 and self.info.get("video_path") is None:
            self.info["video_path"] = DEFAULT_VIDEO_PATH
        chunk = self._episode_chunk(episode_index)
        self.info["total_chunks"] = max(self.info["total_chunks"], chunk + 1)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        write_json(self.info, self.root / INFO_PATH)
        self._cleanup_episode_images(episode_index)
        self.episode_buffer = self.create_episode_buffer()

    def _export_episode_videos(self, episode_index: int) -> int:
        if not self.video_keys or self._video_export_disabled:
            return 0
        created = 0
        for image_key in self.video_keys:
            episode_dir = self.root / "images" / image_key / f"episode_{episode_index:06d}"
            if not episode_dir.is_dir():
                continue
            frame_paths = sorted(episode_dir.glob("frame_*.png"))
            if not frame_paths:
                continue
            video_path = self._get_video_file_path(episode_index, image_key)
            try:
                write_video_from_images(frame_paths, video_path, self.fps)
            except VideoExportError as exc:
                logging.warning("Disabling video export after failure for %s episode %d: %s", image_key, episode_index, exc)
                self._video_export_disabled = True
                return 0
            except Exception:
                logging.exception(
                    "Unexpected error while exporting video for %s episode %d", image_key, episode_index
                )
                continue
            created += 1
        return created

    def finalize(self) -> None:
        if self.stats is not None:
            write_json(serialize_stats(self.stats), self.root / STATS_PATH)


@dataclass
class KeySpec:
    source: str
    target: str
    dtype: str
    shape: tuple[int, ...]
    kind: str

    def format_value(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if self.dtype == "float32":
            return arr.astype(np.float32, copy=False)
        if self.dtype == "int64":
            return arr.astype(np.int64, copy=False)
        if self.dtype == "bool":
            return arr.astype(np.bool_, copy=False)
        return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a diffusion-policy ReplayBuffer to a LeRobot dataset")
    parser.add_argument("--input", required=True, help="Replay buffer directory or .zarr.zip file")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the dataset")
    parser.add_argument("--repo-id", required=True, help="Dataset identifier (metadata only)")
    parser.add_argument("--fps", type=int, default=10, help="Dataset frame rate")
    parser.add_argument("--task", default="task", help="Task description to attach to every frame")
    parser.add_argument("--robot-type", default=None, help="Optional robot type stored in metadata")
    parser.add_argument("--action-key", default=None, help="Replay buffer key containing actions")
    parser.add_argument("--target-action-key", default="action", help="Action key name for output dataset")
    parser.add_argument("--timestamp-key", default="timestamp", help="Key containing timestamps (optional)")
    parser.add_argument("--include-keys", nargs="+", default=None, help="Subset of keys to export")
    parser.add_argument("--exclude-keys", nargs="+", default=None, help="Keys to ignore")
    parser.add_argument("--image-keys", nargs="+", default=None, help="Output keys treated as images")
    parser.add_argument("--key-map", nargs="+", default=None, help="Rename directives source:target")
    parser.add_argument("--start-episode", type=int, default=0, help="First episode index to export")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to export")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


@contextlib.contextmanager
def replay_buffer_from_path(path: str | Path) -> DiffusionReplayBuffer:
    store = None
    try:
        resolved = Path(path).expanduser()
        if resolved.is_dir():
            root = zarr.open_group(str(resolved), mode="r")
        elif resolved.suffix in {".zip", ".zarr.zip"}:
            store = zarr.ZipStore(str(resolved), mode="r")
            root = zarr.group(store=store)
        else:
            root = zarr.open_group(str(resolved), mode="r")
        yield DiffusionReplayBuffer(root)
    finally:
        if store is not None:
            store.close()


def parse_key_map(entries: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not entries:
        return mapping
    for raw in entries:
        if ":" not in raw:
            raise ValueError(f"Invalid key mapping '{raw}'")
        source, target = raw.split(":", 1)
        mapping[source] = target
    return mapping


def infer_action_key(explicit: str | None, replay_buffer: DiffusionReplayBuffer) -> str:
    if explicit:
        if explicit not in replay_buffer:
            raise KeyError(f"Action key '{explicit}' not found in replay buffer")
        return explicit
    for candidate in ("action", "actions", "action_2d"):
        if candidate in replay_buffer:
            return candidate
    raise KeyError("Unable to infer action key; please provide --action-key")


def infer_dtype(array: zarr.Array) -> str:
    if np.issubdtype(array.dtype, np.bool_):
        return "bool"
    if np.issubdtype(array.dtype, np.integer):
        return "int64"
    return "float32"


def is_image_array(arr: zarr.Array, explicit: set[str] | None, target_name: str) -> bool:
    if explicit is not None:
        return target_name in explicit
    if arr.ndim != 4:
        return False
    if arr.shape[-1] not in (1, 3, 4):
        return False
    return np.issubdtype(arr.dtype, np.uint8)


def build_key_specs(
    replay_buffer: DiffusionReplayBuffer,
    keys: list[str],
    image_targets: set[str] | None,
    key_map: dict[str, str],
    action_source: str,
    action_target: str,
) -> list[KeySpec]:
    specs = []
    seen_targets: set[str] = set()
    for source in keys:
        if source not in replay_buffer:
            raise KeyError(f"Replay buffer missing key '{source}'")
        target = key_map.get(source, source)
        if source == action_source:
            target = action_target
        if target in seen_targets:
            raise ValueError(f"Target key '{target}' defined multiple times")
        seen_targets.add(target)
        arr = replay_buffer[source]
        frame_shape = tuple(arr.shape[1:]) if arr.ndim > 1 else (1,)
        if is_image_array(arr, image_targets, target):
            dtype = "image"
            kind = "image"
        else:
            dtype = infer_dtype(arr)
            kind = "array"
        specs.append(KeySpec(source=source, target=target, dtype=dtype, shape=frame_shape, kind=kind))
    return specs


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)


def resolve_timestamp_key(timestamp_key: str | None, replay_buffer: DiffusionReplayBuffer) -> str | None:
    if timestamp_key and timestamp_key in replay_buffer:
        return timestamp_key
    return None


def select_keys(
    replay_buffer: DiffusionReplayBuffer,
    include: list[str] | None,
    exclude: list[str] | None,
    action_key: str,
    timestamp_key: str | None,
) -> list[str]:
    available = list(replay_buffer.keys())
    selected = include if include is not None else available
    excluded = set(exclude or [])
    if timestamp_key is not None:
        excluded.add(timestamp_key)
    filtered = [key for key in selected if key not in excluded]
    if action_key not in filtered:
        filtered.append(action_key)
    return filtered


def gather_features(specs: list[KeySpec]) -> dict[str, dict]:
    features = {}
    for spec in specs:
        features[spec.target] = {"dtype": spec.dtype, "shape": spec.shape, "names": None}
    return features


def convert_episode(
    dataset: SimpleLeRobotDataset,
    specs: list[KeySpec],
    replay_buffer: DiffusionReplayBuffer,
    start: int,
    end: int,
    task_name: str,
    timestamp_key: str | None,
) -> int:
    length = end - start
    if length <= 0:
        return 0
    episode_data = {spec.source: np.asarray(replay_buffer[spec.source][start:end]) for spec in specs}
    timestamps = None
    if timestamp_key is not None:
        timestamps = np.asarray(replay_buffer[timestamp_key][start:end])
    for idx in range(length):
        frame = {"task": task_name}
        if timestamps is not None:
            frame["timestamp"] = np.array([timestamps[idx]], dtype=np.float32)
        for spec in specs:
            value = episode_data[spec.source][idx]
            if spec.kind == "image":
                frame[spec.target] = value.astype(np.uint8)
            else:
                frame[spec.target] = spec.format_value(value)
        dataset.add_frame(frame)
    dataset.save_episode()
    return length


def convert_replay_buffer(args: argparse.Namespace) -> None:
    output_path = Path(args.output_dir).expanduser()
    ensure_output_dir(output_path, args.overwrite)
    key_map = parse_key_map(args.key_map)
    image_targets = set(args.image_keys) if args.image_keys else None
    with replay_buffer_from_path(args.input) as replay_buffer:
        action_source = infer_action_key(args.action_key, replay_buffer)
        timestamp_key = resolve_timestamp_key(args.timestamp_key, replay_buffer)
        selected_keys = select_keys(replay_buffer, args.include_keys, args.exclude_keys, action_source, timestamp_key)
        specs = build_key_specs(
            replay_buffer=replay_buffer,
            keys=selected_keys,
            image_targets=image_targets,
            key_map=key_map,
            action_source=action_source,
            action_target=args.target_action_key,
        )
        feature_dict = gather_features(specs)
        dataset = SimpleLeRobotDataset(
            repo_id=args.repo_id,
            root=output_path,
            fps=args.fps,
            features=feature_dict,
            robot_type=args.robot_type,
        )
        episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
        starts = np.concatenate(([0], episode_ends[:-1])) if len(episode_ends) else np.array([], dtype=np.int64)
        processed = 0
        converted_frames = 0
        for idx, (start, end) in enumerate(zip(starts, episode_ends)):
            if idx < args.start_episode:
                continue
            if args.max_episodes is not None and processed >= args.max_episodes:
                break
            frames = convert_episode(dataset, specs, replay_buffer, int(start), int(end), args.task, timestamp_key)
            processed += 1
            converted_frames += frames
            logging.info("Converted episode %d with %d frames", idx, frames)
        dataset.finalize()
        logging.info("Finished conversion: %d episodes, %d frames", processed, converted_frames)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    convert_replay_buffer(args)


if __name__ == "__main__":
    main()
