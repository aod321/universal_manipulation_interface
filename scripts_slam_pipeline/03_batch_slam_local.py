"""
本地运行版本：批量SLAM
python scripts_slam_pipeline/03_batch_slam_local.py -i <input_dir> [--orb_slam3_dir <path>] [--setting_file <path>]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask


# %%
def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('--orb_slam3_dir', default=None, help='Path to ORB_SLAM3 directory (default: ../orb_slam3_code)')
@click.option('--setting_file', default=None, help='Path to setting YAML file (default: gopro13_maxlens_fisheye_setting.yaml)')
@click.option('-n', '--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('--slam_threads', type=int, default=None, help='Number of threads per SLAM process (default: 4)')
@click.option('--use_processes', is_flag=True, default=False, help='Use ProcessPoolExecutor instead of ThreadPoolExecutor (better for CPU-intensive tasks)')
def main(input_dir, map_path, orb_slam3_dir, setting_file, num_workers, max_lost_frames, timeout_multiple, slam_threads, use_processes):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file(), f"Map file not found: {map_path}"

    if num_workers is None:
        # Use all CPU cores by default for better performance
        num_workers = multiprocessing.cpu_count()
    
    if slam_threads is None:
        # Default threads per SLAM process
        slam_threads = 4
    
    print(f"Using {num_workers} parallel workers")
    print(f"Each SLAM process will use {slam_threads} threads")
    print(f"Total video directories to process: {len(input_video_dirs)}")
    
    # Count videos that need processing
    videos_to_process = [d for d in input_video_dirs if not d.joinpath('camera_trajectory.csv').is_file()]
    print(f"Videos that need processing: {len(videos_to_process)}")

    # Setup ORB_SLAM3 paths
    if orb_slam3_dir is None:
        orb_slam3_dir = pathlib.Path(__file__).parent.parent.joinpath('orb_slam3_code')
    else:
        orb_slam3_dir = pathlib.Path(os.path.expanduser(orb_slam3_dir)).absolute()
    
    assert orb_slam3_dir.is_dir(), f"ORB_SLAM3 directory not found: {orb_slam3_dir}"
    
    gopro_slam_exe = orb_slam3_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro_slam')
    vocabulary_path = orb_slam3_dir.joinpath('Vocabulary', 'ORBvoc.txt')
    
    if setting_file is None:
        setting_file = orb_slam3_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro13_maxlens_fisheye_setting.yaml')
    else:
        setting_file = pathlib.Path(os.path.expanduser(setting_file)).absolute()
    
    assert gopro_slam_exe.is_file(), f"gopro_slam executable not found: {gopro_slam_exe}. Please build ORB_SLAM3 first."
    assert vocabulary_path.is_file(), f"Vocabulary file not found: {vocabulary_path}"
    assert setting_file.is_file(), f"Setting file not found: {setting_file}"

    # Note: Even with ThreadPoolExecutor, subprocess.run() will spawn separate processes
    # You should see multiple 'gopro_slam' processes in htop, not Python processes
    # To see threads in htop, press 'H' key to toggle thread view
    with tqdm(total=len(input_video_dirs)) as pbar:
        # Use ProcessPoolExecutor for CPU-intensive tasks, ThreadPoolExecutor for I/O-bound tasks
        executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
        executor_name = "ProcessPoolExecutor" if use_processes else "ThreadPoolExecutor"
        print(f"Using {executor_name} with {num_workers} workers")
        print("Note: Check for 'gopro_slam' processes in htop (not Python processes)")
        with executor_class(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # Prepare paths for local execution
                csv_path = video_dir.joinpath('camera_trajectory.csv')
                video_path = video_dir.joinpath('raw_video.mp4')
                json_path = video_dir.joinpath('imu_data.json')
                mask_path = video_dir.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
                slam_mask = draw_predefined_mask(
                    slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                # Build command for local execution
                cmd = [
                    str(gopro_slam_exe),
                    '--vocabulary', str(vocabulary_path),
                    '--setting', str(setting_file),
                    '--input_video', str(video_path),
                    '--input_imu_json', str(json_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_path),
                    '--mask_img', str(mask_path),
                    '--max_lost_frames', str(max_lost_frames),
                    '--num_threads', str(slam_threads)
                ]

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                future = executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout)
                futures.add(future)
                print(f"Submitted task for {video_dir.name} (total active: {len(futures)})")

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done! Result:")
    print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()

