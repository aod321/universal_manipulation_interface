#!/bin/bash
# 验证今天建图的命令（data1110）
# 注意：需要先确保mask文件尺寸正确（3840x3360）

cd /home/yinzi/universal_manipulation_interface/orb_slam3_code/Examples/Monocular-Inertial

./gopro_slam \
  --vocabulary /home/yinzi/universal_manipulation_interface/orb_slam3_code/Vocabulary/ORBvoc.txt \
  --setting /home/yinzi/universal_manipulation_interface/orb_slam3_code/Examples/Monocular-Inertial/gopro13_maxlens_fisheye_setting.yaml \
  --input_video /home/yinzi/universal_manipulation_interface/data1110/demos/mapping/raw_video.mp4 \
  --input_imu_json /home/yinzi/universal_manipulation_interface/data1110/demos/mapping/imu_data.json \
  --output_trajectory_csv /home/yinzi/universal_manipulation_interface/data1110/demos/mapping/new_traj.csv \
  --save_map /home/yinzi/universal_manipulation_interface/data1110/demos/mapping/map_atlas_new.osa \
  --mask_img /home/yinzi/universal_manipulation_interface/data1110/demos/mapping/slam_mask.png \
  -g

