"""
本地运行版本：创建地图
python scripts_slam_pipeline/02_create_map_local.py -i <input_dir> [--orb_slam3_dir <path>] [--setting_file <path>]
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
import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('--orb_slam3_dir', default=None, help='Path to ORB_SLAM3 directory (default: ../orb_slam3_code)')
@click.option('--setting_file', default=None, help='Path to setting YAML file (default: gopro13SN7674_maxlens_fisheye_setting_960x840.yaml)')
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, orb_slam3_dir, setting_file, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup ORB_SLAM3 paths
    if orb_slam3_dir is None:
        orb_slam3_dir = pathlib.Path(__file__).parent.parent.joinpath('orb_slam3_code')
    else:
        orb_slam3_dir = pathlib.Path(os.path.expanduser(orb_slam3_dir)).absolute()
    
    assert orb_slam3_dir.is_dir(), f"ORB_SLAM3 directory not found: {orb_slam3_dir}"
    
    gopro_slam_exe = orb_slam3_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro_slam')
    vocabulary_path = orb_slam3_dir.joinpath('Vocabulary', 'ORBvoc.txt')
    
    if setting_file is None:
        # setting_file = orb_slam3_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro13_maxlens_fisheye_setting.yaml')
        setting_file = orb_slam3_dir.joinpath('Examples', 'Monocular-Inertial', 'gopro13SN7674_maxlens_fisheye_setting_960x840.yaml')
    else:
        setting_file = pathlib.Path(os.path.expanduser(setting_file)).absolute()
    
    assert gopro_slam_exe.is_file(), f"gopro_slam executable not found: {gopro_slam_exe}. Please build ORB_SLAM3 first."
    assert vocabulary_path.is_file(), f"Vocabulary file not found: {vocabulary_path}"
    assert setting_file.is_file(), f"Setting file not found: {setting_file}"

    # Prepare paths
    csv_path = video_dir.joinpath('mapping_camera_trajectory.csv')
    video_path = video_dir.joinpath('raw_video.mp4')
    json_path = video_dir.joinpath('imu_data.json')
    mask_path = video_dir.joinpath('slam_mask.png')
    
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
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
        '--save_map', str(map_path)
    ]
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)
    if result.returncode != 0:
        print(f"Error: SLAM failed with return code {result.returncode}")
        print(f"Check {stdout_path} and {stderr_path} for details")


# %%
if __name__ == "__main__":
    main()
