"""
Visualize camera trajectory files
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

def visualize_trajectories(base_dir):
    """Visualize all camera trajectory files"""
    base_path = Path(base_dir)
    demos_dir = base_path / 'demos'
    
    # Find all camera_trajectory.csv files, excluding mapping
    all_trajectory_files = sorted(list(demos_dir.glob('**/camera_trajectory.csv')))
    trajectory_files = [f for f in all_trajectory_files if 'mapping' not in str(f)]
    
    if len(trajectory_files) == 0:
        print("No camera_trajectory.csv files found")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color list
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_files)))
    
    # Store all trajectory bounds for axis setting
    all_x, all_y, all_z = [], [], []
    
    # Plot each trajectory
    for idx, traj_file in enumerate(trajectory_files):
        try:
            # Read CSV file
            df = pd.read_csv(traj_file)
            
            # Extract trajectory name (from directory name)
            traj_name = traj_file.parent.name
            
            # Extract position data
            x = df['x'].values
            y = df['y'].values
            z = df['z'].values
            is_lost = df['is_lost'].values if 'is_lost' in df.columns else np.zeros(len(x), dtype=bool)
            
            # Filter out lost frames
            valid_mask = ~is_lost
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            
            # Plot trajectory
            ax.plot(x_valid, y_valid, z_valid, 
                   color=colors[idx], 
                   label=traj_name, 
                   linewidth=1.5, 
                   alpha=0.7)
            
            # Mark start and end points
            if len(x_valid) > 0:
                ax.scatter(x_valid[0], y_valid[0], z_valid[0], 
                          color=colors[idx], 
                          marker='o', s=100, 
                          label=f'{traj_name} start',
                          edgecolors='black', linewidths=1)
                ax.scatter(x_valid[-1], y_valid[-1], z_valid[-1], 
                          color=colors[idx], 
                          marker='s', s=100, 
                          label=f'{traj_name} end',
                          edgecolors='black', linewidths=1)
            
            # Collect boundary data
            all_x.extend(x_valid)
            all_y.extend(y_valid)
            all_z.extend(z_valid)
            
            print(f"  {traj_name}: {len(x_valid)} valid points")
            
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            continue
    
    # Set axis labels
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Camera Trajectories Visualization', fontsize=14, fontweight='bold')
    
    # Set axis ranges (with margin)
    if all_x and all_y and all_z:
        margin = 0.1
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        z_range = max(all_z) - min(all_z)
        max_range = max(x_range, y_range, z_range) * 0.5
        
        x_center = (max(all_x) + min(all_x)) / 2
        y_center = (max(all_y) + min(all_y)) / 2
        z_center = (max(all_z) + min(all_z)) / 2
        
        ax.set_xlim([x_center - max_range, x_center + max_range])
        ax.set_ylim([y_center - max_range, y_center + max_range])
        ax.set_zlim([z_center - max_range, z_center + max_range])
    
    # Add legend (only trajectory names, no start/end points)
    handles, labels = ax.get_legend_handles_labels()
    # Filter out start/end point labels
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if 'start' not in label and 'end' not in label:
            unique_labels[label] = handle
    
    ax.legend(list(unique_labels.values()), list(unique_labels.keys()), 
             loc='upper left', fontsize=9, ncol=1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set viewpoint
    ax.view_init(elev=20, azim=45)
    
    # Save image
    output_path = base_path / 'trajectories_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show figure
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = 'data1103'
    
    visualize_trajectories(base_dir)

