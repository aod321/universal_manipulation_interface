# ReplayBuffer to LeRobot dataset converter

`scripts/replaybuffer_to_lerobot.py` ports a diffusion-policy style ReplayBuffer (stored as a zarr directory or zip archive) to the LeRobot dataset format. It does **not** depend on the upstream `lerobot` repository—the script includes the minimal dataset writer that mirrors the LeRobot format.

## Requirements

- Python packages: `pyarrow`, `numpy`, `zarr`, and `Pillow`. The optional `datasets` dependency unlocks Hugging Face's Dataset writer; if it is missing, the converter falls back to the built-in minimal dataset implementation.
- Image modalities are written as PNG-backed `datasets.Image` columns (or plain file paths when using the fallback). The script now also exports per-episode MP4 files for every image key under `videos/chunk-XXX/<image_key>/episode_XXXXXX.mp4`, matching the layout used by `real/visualize_dataset.py`. MP4 export requires `opencv-python`; if `cv2` is not available the converter logs a warning and skips video generation.
- Input replay buffer accessible as a folder or `.zarr.zip` archive. Archives encoded with `deflate64` need to be unzipped manually because Python's `zipfile` module cannot stream them.

## Usage

```bash
python scripts/replaybuffer_to_lerobot.py \
  --input /path/to/replay_buffer \
  --output-dir /tmp/push_cube_lerobot \
  --repo-id local/push_cube_dataset \
  --fps 20 \
  --task "Push Cube" \
  --action-key action_2d \
  --target-action-key action \
  --timestamp-key timestamp \
  --image-keys camera0_rgb camera1_rgb
```

Notable options:

- `--include-keys/--exclude-keys` filter which ReplayBuffer entries are exported.
- `--key-map source:target` renames observation keys (e.g., `robot0_eef_pos:observation.eef_pos`).
- `--image-keys` marks the LeRobot field names that should be encoded as frames/videos. Other keys are treated as float/int arrays.
- `--start-episode`, `--max-episodes` let you export a subset of the data for quick smoke tests.
- Image modalities default to channel-last arrays; use `--image-keys` to mark keys that should be saved as images if auto-detection fails.

Each frame stored in the LeRobot dataset automatically receives the task description string, and timestamps are copied when `--timestamp-key` is present. The converter enforces that the action tensor exists (or you can point `--action-key` to the right ReplayBuffer field such as `action_2d`).

The generated MP4s share the same chunking scheme as the parquet files, which allows `/home/yinzi/dummy_ctrl/real/visualize_dataset.py` (or similar rerun-based viewers) to visualize `/home/yinzi/lerobot_dataset_gopro` outputs directly.
