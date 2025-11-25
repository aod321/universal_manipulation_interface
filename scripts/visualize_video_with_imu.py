#!/usr/bin/env python3
"""Stream a camera video and IMU signals together in Rerun."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import math

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "opencv-python is required for video playback. Install it to run this script."
    ) from exc


AXIS_LABELS = ["x", "y", "z", "w"]
GRAVITY = 9.80665
ACC_MAG_EPS = 0.8
GYRO_MAG_EPS = 0.2


@dataclass
class ImuEvent:
    """A single IMU measurement annotated with metadata for logging."""

    timestamp: float
    stream: str
    values: Sequence[float]
    axis_labels: Sequence[str]
    units: str | None
    temperature_c: float | None


@dataclass
class TimedSample:
    timestamp: float
    value: np.ndarray


class SampleSeries:
    def __init__(self, samples: Sequence[TimedSample]):
        if not samples:
            raise ValueError("Cannot create SampleSeries without samples")
        ordered = sorted(samples, key=lambda sample: sample.timestamp)
        self.timestamps = np.array([sample.timestamp for sample in ordered], dtype=float)
        self.values = np.array([sample.value for sample in ordered], dtype=float)

    def value_at(self, timestamp: float) -> np.ndarray:
        if timestamp <= self.timestamps[0]:
            return self.values[0]
        if timestamp >= self.timestamps[-1]:
            return self.values[-1]
        idx = int(np.searchsorted(self.timestamps, timestamp))
        t0 = self.timestamps[idx - 1]
        t1 = self.timestamps[idx]
        if t1 == t0:
            return self.values[idx].copy()
        ratio = (timestamp - t0) / (t1 - t0)
        return self.values[idx - 1] * (1.0 - ratio) + self.values[idx] * ratio


class ConstantAccelerationKalman:
    def __init__(
        self,
        process_var: float = 1e-2,
        acceleration_var: float = 0.5,
        velocity_var: float = 1e-2,
    ):
        self.state = np.zeros(3)
        self.cov = np.eye(3)
        self.process_var = process_var
        self.acceleration_var = acceleration_var
        self.velocity_var = velocity_var

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        F = np.array(
            [
                [1.0, dt, 0.5 * dt2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ]
        )
        q = self.process_var
        Q = q * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ]
        )
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q

    def _update(self, measurement: float, H: np.ndarray, var: float) -> None:
        z = np.array([[measurement]])
        S = H @ self.cov @ H.T + var
        K = self.cov @ H.T / S
        y = z - H @ self.state
        self.state = self.state + (K.flatten() * y.flatten())
        I = np.eye(3)
        self.cov = (I - K @ H) @ self.cov

    def update_acceleration(self, measurement: float) -> None:
        H = np.array([[0.0, 0.0, 1.0]])
        self._update(measurement, H, self.acceleration_var)

    def update_velocity(self, measurement: float = 0.0) -> None:
        H = np.array([[0.0, 1.0, 0.0]])
        self._update(measurement, H, self.velocity_var)


class AngleRateKalman:
    def __init__(self, process_var: float = 1e-3, angle_var: float = 1e-2, rate_var: float = 5e-3):
        self.state = np.zeros(2)
        self.cov = np.eye(2)
        self.process_var = process_var
        self.angle_var = angle_var
        self.rate_var = rate_var

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = q * np.array([[dt3 / 3.0, dt2 / 2.0], [dt2 / 2.0, dt]])
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q

    def update(self, angle_measure: float | None, rate_measure: float | None) -> None:
        measurements: List[tuple[np.ndarray, np.ndarray, float]] = []
        if angle_measure is not None:
            measurements.append((np.array([[angle_measure]]), np.array([[1.0, 0.0]]), self.angle_var))
        if rate_measure is not None:
            measurements.append((np.array([[rate_measure]]), np.array([[0.0, 1.0]]), self.rate_var))
        for z, H, R_var in measurements:
            S = H @ self.cov @ H.T + R_var
            K = self.cov @ H.T / S
            y = z - H @ self.state
            self.state = self.state + (K.flatten() * y.flatten())
            I = np.eye(2)
            self.cov = (I - K @ H) @ self.cov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path(
            "/home/yinzi/universal_manipulation_interface/data1118/demos/mapping/raw_video.mp4"
        ),
        help="Path to the MP4 file to display.",
    )
    parser.add_argument(
        "--imu-path",
        type=Path,
        default=Path(
            "/home/yinzi/universal_manipulation_interface/data1118/demos/mapping/imu_data.json"
        ),
        help="Path to the IMU JSON dump (see `imu_signals` example).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional name for the Rerun run. Defaults to the video file stem.",
    )
    parser.add_argument(
        "--timeline",
        default="time",
        help="Name of the Rerun timeline used for both video and IMU data.",
    )
    parser.add_argument(
        "--video-entity",
        default="world/video",
        help="Entity path under which video frames are logged.",
    )
    parser.add_argument(
        "--imu-entity-root",
        default="world/imu",
        help="Entity path prefix for IMU streams.",
    )
    parser.add_argument(
        "--video-offset",
        type=float,
        default=0.0,
        help="Seconds to shift the video timestamps by (positive delays video).",
    )
    parser.add_argument(
        "--imu-offset",
        type=float,
        default=0.0,
        help="Seconds to shift the IMU timestamps by (positive delays IMU).",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Override the video FPS. If omitted we query the container and fall back to metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of frames to log (useful for debugging).",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Do not spawn a new Rerun Viewer window automatically.",
    )
    parser.set_defaults(spawn=True)
    return parser.parse_args()


def ensure_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def parse_iso_timestamp(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    normalized = (date_str.rstrip("Z") + "+00:00") if date_str.endswith("Z") else date_str
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def infer_axis_labels(length: int) -> Sequence[str]:
    if length <= len(AXIS_LABELS):
        return tuple(AXIS_LABELS[:length])
    return tuple(str(i) for i in range(length))


def load_imu_streams(payload: Dict[str, Any]) -> Dict[str, Any]:
    numeric_keys = sorted(key for key in payload.keys() if key.isdigit())
    if not numeric_keys:
        raise ValueError("IMU payload missing numeric sensor entries")
    key = numeric_keys[0]
    section = payload[key]
    streams = section.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("IMU payload contains no streams")
    return streams


def compute_reference_timestamp(streams: Dict[str, Any]) -> datetime | None:
    reference: datetime | None = None
    for stream in streams.values():
        for sample in stream.get("samples", []):
            ts = parse_iso_timestamp(sample.get("date"))
            if ts is None:
                continue
            if reference is None or ts < reference:
                reference = ts
    return reference


def build_imu_events(
    streams: Dict[str, Any], offset: float, reference: datetime | None = None
) -> List[ImuEvent]:
    reference = reference or compute_reference_timestamp(streams)
    if reference is None:
        raise ValueError("No valid timestamps found in IMU samples")

    events: List[ImuEvent] = []
    for stream_name, stream in streams.items():
        samples = stream.get("samples", [])
        if not samples:
            continue
        units = stream.get("units")
        axis_labels: Sequence[str] | None = None
        for sample in samples:
            ts = parse_iso_timestamp(sample.get("date"))
            if ts is None:
                continue
            values = sample.get("value")
            if values is None:
                continue
            try:
                numeric_values = tuple(float(v) for v in values)
            except (TypeError, ValueError):
                continue
            if axis_labels is None or len(axis_labels) != len(numeric_values):
                axis_labels = infer_axis_labels(len(numeric_values))
            timestamp = (ts - reference).total_seconds() + offset
            temp_value = sample.get("temperature [°C]")
            temperature_c = float(temp_value) if isinstance(temp_value, (int, float)) else None
            events.append(
                ImuEvent(
                    timestamp=timestamp,
                    stream=stream_name,
                    values=numeric_values,
                    axis_labels=axis_labels,
                    units=units,
                    temperature_c=temperature_c,
                )
            )
    events.sort(key=lambda evt: evt.timestamp)
    return events


def build_timed_samples(
    stream: Dict[str, Any], reference: datetime, offset: float
) -> List[TimedSample]:
    samples: List[TimedSample] = []
    for sample in stream.get("samples", []):
        ts = parse_iso_timestamp(sample.get("date"))
        if ts is None:
            continue
        values = sample.get("value")
        if values is None:
            continue
        try:
            numeric = np.array([float(v) for v in values], dtype=float)
        except (TypeError, ValueError):
            continue
        timestamp = (ts - reference).total_seconds() + offset
        samples.append(TimedSample(timestamp=timestamp, value=numeric))
    return samples


def extract_stream_samples(
    streams: Dict[str, Any], name: str, reference: datetime, offset: float
) -> List[TimedSample]:
    section = streams.get(name)
    if not section:
        return []
    return build_timed_samples(section, reference, offset)


def unwrap_angle(value: float, previous: float) -> float:
    delta = value - previous
    while delta > np.pi:
        value -= 2.0 * np.pi
        delta = value - previous
    while delta < -np.pi:
        value += 2.0 * np.pi
        delta = value - previous
    return value


def derive_roll_pitch(accel: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(accel))
    if norm < 1e-6:
        return None
    ax, ay, az = accel / norm
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return np.array([roll, pitch], dtype=float)


def detect_stationary(accel: np.ndarray, gyro: np.ndarray) -> bool:
    acc_norm = float(np.linalg.norm(accel))
    gyro_norm = float(np.linalg.norm(gyro))
    acc_stationary = abs(acc_norm - GRAVITY) < ACC_MAG_EPS
    gyro_stationary = gyro_norm < GYRO_MAG_EPS
    return acc_stationary and gyro_stationary


def build_fused_pose_events(
    streams: Dict[str, Any], offset: float, reference: datetime | None
) -> List[ImuEvent]:
    if reference is None:
        return []
    accel_samples = extract_stream_samples(streams, "ACCL", reference, offset)
    if not accel_samples:
        return []
    gyro_samples = extract_stream_samples(streams, "GYRO", reference, offset)
    gyro_series = SampleSeries(gyro_samples) if gyro_samples else None

    position_filters = [ConstantAccelerationKalman() for _ in range(3)]
    angle_filters = [AngleRateKalman() for _ in range(3)]

    fused_events: List[ImuEvent] = []
    prev_timestamp = accel_samples[0].timestamp
    prev_accel_angles = np.zeros(2)
    linear_bias = np.zeros(3)
    bias_time_constant = 3.0

    for sample in accel_samples:
        timestamp = sample.timestamp
        dt = timestamp - prev_timestamp
        prev_timestamp = timestamp
        for filt in position_filters:
            filt.predict(dt)
        for filt in angle_filters:
            filt.predict(dt)

        gyro_value = (
            gyro_series.value_at(timestamp) if gyro_series else np.zeros(3, dtype=float)
        )
        accel_angles = derive_roll_pitch(sample.value)
        roll_pitch_measure: List[float | None] = [None, None]
        if accel_angles is not None:
            roll_pitch_measure = [
                unwrap_angle(float(accel_angles[0]), float(prev_accel_angles[0])),
                unwrap_angle(float(accel_angles[1]), float(prev_accel_angles[1])),
            ]
            prev_accel_angles = np.array(
                [
                    roll_pitch_measure[0]
                    if roll_pitch_measure[0] is not None
                    else prev_accel_angles[0],
                    roll_pitch_measure[1]
                    if roll_pitch_measure[1] is not None
                    else prev_accel_angles[1],
                ]
            )

        for axis, filt in enumerate(angle_filters):
            angle_measure = None
            if axis < 2 and roll_pitch_measure[axis] is not None:
                angle_measure = float(roll_pitch_measure[axis])
            rate_measure = float(gyro_value[axis]) if axis < len(gyro_value) else None
            filt.update(angle_measure, rate_measure)

        fused_angles = np.array([filt.state[0] for filt in angle_filters], dtype=float)
        fused_rotation = R.from_euler("xyz", fused_angles, degrees=False)
        gravity_body = fused_rotation.inv().apply(np.array([0.0, 0.0, -GRAVITY]))
        linear_body = sample.value - gravity_body
        linear_world = fused_rotation.apply(linear_body)
        stationary = detect_stationary(sample.value, gyro_value)
        if stationary:
            if bias_time_constant <= 0 or dt <= 0:
                alpha = 1.0
            else:
                alpha = min(1.0, dt / bias_time_constant)
            linear_bias = (1.0 - alpha) * linear_bias + alpha * linear_world
        corrected_linear = linear_world - linear_bias

        for axis, filt in enumerate(position_filters):
            filt.update_acceleration(float(corrected_linear[axis]))
            if stationary:
                filt.update_velocity(0.0)

        fused_values = [float(filt.state[0]) for filt in position_filters]
        fused_values.extend(float(angle) for angle in fused_angles)
        fused_events.append(
            ImuEvent(
                timestamp=timestamp,
                stream="fusion",
                values=tuple(fused_values),
                axis_labels=("x", "y", "z", "rpx", "rpy", "rpz"),
                units=None,
                temperature_c=None,
            )
        )

    return fused_events


def log_imu_event(event: ImuEvent, timeline: str, entity_root: str) -> None:
    rr.set_time_seconds(timeline, float(event.timestamp))
    unit_suffix = f" [{event.units}]" if event.units else ""
    for idx, value in enumerate(event.values):
        label = event.axis_labels[idx] if idx < len(event.axis_labels) else str(idx)
        rr.log(
            f"{entity_root}/{event.stream}/{label}",
            rr.TimeSeriesScalar(float(value), label=f"{event.stream} {label}{unit_suffix}"),
        )
    if event.temperature_c is not None:
        rr.log(
            f"{entity_root}/{event.stream}/temperature",
            rr.TimeSeriesScalar(
                float(event.temperature_c), label=f"{event.stream} temperature [°C]"
            ),
        )


def stream_video(
    video_path: Path,
    timeline: str,
    entity: str,
    events: List[ImuEvent],
    video_offset: float,
    fps_override: float | None,
    video_fps_hint: float | None,
    max_frames: int | None,
    imu_entity_root: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or video_fps_hint or 30.0
    if fps <= 0:
        fps = video_fps_hint or 30.0

    next_event_idx = 0
    total_events = len(events)

    def flush_up_to(timestamp: float) -> None:
        nonlocal next_event_idx
        while next_event_idx < total_events and events[next_event_idx].timestamp <= timestamp:
            log_imu_event(events[next_event_idx], timeline, imu_entity_root)
            next_event_idx += 1

    flush_up_to(video_offset)

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            timestamp = (frame_index / fps) + video_offset
            flush_up_to(timestamp)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rr.set_time_seconds(timeline, timestamp)
            rr.log(entity, rr.Image(frame_rgb))
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()

    while next_event_idx < total_events:
        log_imu_event(events[next_event_idx], timeline, imu_entity_root)
        next_event_idx += 1


def main() -> None:
    args = parse_args()
    video_path = ensure_path(args.video_path, "Video file")
    imu_path = ensure_path(args.imu_path, "IMU JSON file")

    with imu_path.open("r", encoding="utf-8") as f:
        imu_payload = json.load(f)
    imu_streams = load_imu_streams(imu_payload)
    reference_timestamp = compute_reference_timestamp(imu_streams)
    imu_events = build_imu_events(imu_streams, args.imu_offset, reference_timestamp)
    fused_events = build_fused_pose_events(imu_streams, args.imu_offset, reference_timestamp)
    if fused_events:
        imu_events.extend(fused_events)
        imu_events.sort(key=lambda evt: evt.timestamp)
    fps_hint = None
    frames_per_second = imu_payload.get("frames/second")
    if isinstance(frames_per_second, (int, float)):
        fps_hint = float(frames_per_second)

    run_name = args.run_name or f"video_imu::{video_path.stem}"
    rr.init(run_name, spawn=args.spawn)
    stream_video(
        video_path=video_path,
        timeline=args.timeline,
        entity=args.video_entity,
        events=imu_events,
        video_offset=args.video_offset,
        fps_override=args.video_fps,
        video_fps_hint=fps_hint,
        max_frames=args.max_frames,
        imu_entity_root=args.imu_entity_root,
    )


if __name__ == "__main__":
    main()
