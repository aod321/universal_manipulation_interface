"""
Visualize mapping trajectory and all demo trajectories together
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

def visualize_map_trajectories(base_dir):
    """Visualize mapping trajectory with all demo trajectories"""
    base_path = Path(base_dir)
    demos_dir = base_path / 'demos'
    
    # Find mapping trajectory
    mapping_traj_file = demos_dir / 'mapping' / 'camera_trajectory.csv'
    
    # Find all demo trajectories (excluding mapping)
    all_trajectory_files = sorted(list(demos_dir.glob('demo*/camera_trajectory.csv')))
    
    if not mapping_traj_file.exists():
        print(f"Warning: Mapping trajectory not found at {mapping_traj_file}")
        print("Will only visualize demo trajectories")
    
    print(f"Found mapping trajectory: {mapping_traj_file.exists()}")
    print(f"Found {len(all_trajectory_files)} demo trajectories")
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    all_x, all_y, all_z = [], [], []
    
    # Plot mapping trajectory first (thicker, different color)
    if mapping_traj_file.exists():
        try:
            df_map = pd.read_csv(mapping_traj_file)
            x_map = df_map['x'].values
            y_map = df_map['y'].values
            z_map = df_map['z'].values
            is_lost_map = df_map['is_lost'].values if 'is_lost' in df_map.columns else np.zeros(len(x_map), dtype=bool)
            
            valid_mask_map = ~is_lost_map
            x_map_valid = x_map[valid_mask_map]
            y_map_valid = y_map[valid_mask_map]
            z_map_valid = z_map[valid_mask_map]
            
            # Plot mapping trajectory in bold red
            ax.plot(x_map_valid, y_map_valid, z_map_valid, 
                   color='red', 
                   label='Mapping Trajectory', 
                   linewidth=3.0, 
                   alpha=0.8)
            
            # Mark start and end
            if len(x_map_valid) > 0:
                ax.scatter(x_map_valid[0], y_map_valid[0], z_map_valid[0], 
                          color='darkred', marker='*', s=200, 
                          label='Mapping Start', edgecolors='black', linewidths=1.5)
                ax.scatter(x_map_valid[-1], y_map_valid[-1], z_map_valid[-1], 
                          color='darkred', marker='*', s=200, 
                          label='Mapping End', edgecolors='black', linewidths=1.5)
            
            all_x.extend(x_map_valid)
            all_y.extend(y_map_valid)
            all_z.extend(z_map_valid)
            
            print(f"  Mapping: {len(x_map_valid)} valid points")
        except Exception as e:
            print(f"Error processing mapping trajectory: {e}")
    
    # Plot demo trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trajectory_files)))
    
    for idx, traj_file in enumerate(all_trajectory_files):
        try:
            df = pd.read_csv(traj_file)
            traj_name = traj_file.parent.name
            
            x = df['x'].values
            y = df['y'].values
            z = df['z'].values
            is_lost = df['is_lost'].values if 'is_lost' in df.columns else np.zeros(len(x), dtype=bool)
            
            valid_mask = ~is_lost
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            
            # Plot demo trajectory
            ax.plot(x_valid, y_valid, z_valid, 
                   color=colors[idx], 
                   label=traj_name, 
                   linewidth=1.5, 
                   alpha=0.6)
            
            # Mark start and end
            if len(x_valid) > 0:
                ax.scatter(x_valid[0], y_valid[0], z_valid[0], 
                          color=colors[idx], marker='o', s=50, 
                          edgecolors='black', linewidths=0.5)
                ax.scatter(x_valid[-1], y_valid[-1], z_valid[-1], 
                          color=colors[idx], marker='s', s=50, 
                          edgecolors='black', linewidths=0.5)
            
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
    ax.set_title('Map Trajectory with Demo Trajectories', fontsize=14, fontweight='bold')
    
    # Set axis ranges
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
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter: keep mapping start/end, remove demo start/end
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if 'Start' in label or 'End' in label or 'Mapping' in label or label.startswith('demo_'):
            if label.startswith('demo_'):
                filtered_handles.append(handle)
                filtered_labels.append(label)
            elif 'Mapping' in label:
                filtered_handles.append(handle)
                filtered_labels.append(label)
    
    ax.legend(filtered_handles, filtered_labels, 
             loc='upper left', fontsize=8, ncol=2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set viewpoint
    ax.view_init(elev=20, azim=45)
    
    # Save image
    output_path = base_path / 'map_trajectories_visualization.png'
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
    
    visualize_map_trajectories(base_dir)





