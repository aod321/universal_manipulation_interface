"""
测试mask效果的工具
从视频中提取一帧，应用mask，查看效果
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import cv2
import numpy as np
from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input', required=True, help='Input video file or image file')
@click.option('-o', '--output', default=None, help='Output image file (default: mask_test_result.jpg)')
@click.option('--frame_idx', type=int, default=0, help='Frame index to extract from video (default: 0)')
@click.option('--height', type=int, default=3360, help='Image height (default: 3360)')
@click.option('--width', type=int, default=None, help='Image width (default: height * 8/7 for 8:7 ratio)')
@click.option('--mirror', is_flag=True, default=True, help='Mask mirror regions')
@click.option('--gripper', is_flag=True, default=False, help='Mask gripper regions')
@click.option('--finger', is_flag=True, default=True, help='Mask finger regions')
@click.option('--show', is_flag=True, default=False, help='Show result in window')
def main(input, output, frame_idx, height, width, mirror, gripper, finger, show):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    assert input_path.is_file(), f"Input file not found: {input_path}"
    
    # Calculate width if not provided (8:7 ratio)
    if width is None:
        width = int(height * 8 / 7)  # 3360 * 8/7 = 3840, but user said 8:7, so height*8/7
        # Actually, if height is 3360 and ratio is 8:7, then width = 3360 * 8/7 = 3840
        # But user said resolution is 3360 with 8:7, which means width:height = 8:7
        # So if height=3360, then width = 3360 * 8/7 = 3840
        # Wait, let me re-read: "我是3360 8:7的" - this could mean width is 3360, ratio is 8:7
        # If width:height = 8:7 and width=3360, then height = 3360 * 7/8 = 2940
        # Let me assume user means height=3360, width=2940 (8:7 means width:height = 8:7? No, usually it's width:height)
        # Actually, 8:7 usually means width:height, so if height=3360, width should be 3360*8/7=3840
        # But user might mean the other way. Let me check: "3360 8:7" - could be 3360x2940 (if 8:7 is width:height and 3360 is height)
        # Or 3840x3360 (if 8:7 is width:height and 3360 is width)
        # Based on context, I think user means height=3360, and 8:7 ratio means width:height=8:7, so width=3840
        # But wait, user said "8:7的" which in Chinese usually means aspect ratio width:height
        # If it's 8:7 and one dimension is 3360, and it's height, then width = 3360 * 8/7 = 3840
        # But if 3360 is width, then height = 3360 * 7/8 = 2940
        # Let me assume the default height=3360 means image height, and 8:7 means width:height
        # So width = 3360 * 8/7 = 3840
        # But user might have meant the opposite. Let me just use the calculation and let user override
    
    print(f"Target resolution: {width}x{height}")
    
    # Load image or extract frame from video
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Load image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Error: Could not load image from {input_path}")
            return
        print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")
    else:
        # Extract frame from video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video has {total_frames} frames")
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not read frame {frame_idx}")
            return
        print(f"Extracted frame {frame_idx}: {img.shape[1]}x{img.shape[0]}")
    
    # Resize if needed
    if img.shape[0] != height or img.shape[1] != width:
        print(f"Resizing from {img.shape[1]}x{img.shape[0]} to {width}x{height}")
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Create mask
    print(f"Creating mask with: mirror={mirror}, gripper={gripper}, finger={finger}")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = draw_predefined_mask(mask, color=255, mirror=mirror, gripper=gripper, finger=finger)
    
    # Apply mask to image
    img_masked = img.copy()
    img_masked[mask > 0] = [0, 0, 0]  # Set masked regions to black
    
    # Create side-by-side comparison
    # Resize for display if too large
    display_scale = min(1920 / width, 1080 / height, 1.0)
    if display_scale < 1.0:
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        img_display = cv2.resize(img, (display_width, display_height))
        img_masked_display = cv2.resize(img_masked, (display_width, display_height))
        mask_display = cv2.resize(mask, (display_width, display_height))
    else:
        img_display = img
        img_masked_display = img_masked
        mask_display = mask
    
    # Convert mask to 3-channel for display
    mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
    
    # Create comparison image
    comparison = np.hstack([img_display, img_masked_display, mask_colored])
    
    # Save result
    if output is None:
        output = input_path.parent.joinpath('mask_test_result.jpg')
    else:
        output = pathlib.Path(os.path.expanduser(output))
    
    # Save full resolution masked image
    cv2.imwrite(str(output), img_masked)
    print(f"Saved masked image to: {output}")
    
    # Save comparison
    comparison_path = output.parent.joinpath(output.stem + '_comparison.jpg')
    cv2.imwrite(str(comparison_path), comparison)
    print(f"Saved comparison to: {comparison_path}")
    
    # Save mask separately
    mask_path = output.parent.joinpath(output.stem + '_mask.png')
    cv2.imwrite(str(mask_path), mask)
    print(f"Saved mask to: {mask_path}")
    
    # Show if requested
    if show:
        cv2.namedWindow('Mask Test Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Mask Test Result', comparison)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# %%
if __name__ == "__main__":
    main()

