#!/usr/bin/env python3
"""
Script to visualize RGB images, depth maps, and point clouds from zarr files.
"""

import os
import argparse
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
try:
    import plotly.graph_objs as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Using matplotlib for 3D visualization.")


def load_zarr_data(zarr_path, load_pointcloud=True):
    """Load RGB images, depth maps, and optionally point clouds from zarr file."""
    print(f"Loading zarr file from: {zarr_path}")
    
    # Open zarr file
    root = zarr.open(zarr_path, mode='r')
    
    # Access data arrays
    if 'data' not in root:
        raise ValueError(f"No 'data' group found in zarr file at {zarr_path}")
    
    data_group = root['data']
    
    # Load RGB images
    if 'img' not in data_group:
        raise ValueError(f"No 'img' dataset found in zarr file")
    img_array = data_group['img']
    print(f"RGB images shape: {img_array.shape}, dtype: {img_array.dtype}")
    
    # Load depth maps
    if 'depth' not in data_group:
        raise ValueError(f"No 'depth' dataset found in zarr file")
    depth_array = data_group['depth']
    print(f"Depth maps shape: {depth_array.shape}, dtype: {depth_array.dtype}")
    
    # Load point clouds (optional)
    pointcloud_array = None
    if load_pointcloud and 'point_cloud' in data_group:
        pointcloud_array = data_group['point_cloud']
        print(f"Point clouds shape: {pointcloud_array.shape}, dtype: {pointcloud_array.dtype}")
    elif load_pointcloud:
        print("Warning: No 'point_cloud' dataset found in zarr file")
    
    # Get number of samples
    num_samples = img_array.shape[0]
    print(f"Total number of samples: {num_samples}")
    
    return img_array, depth_array, pointcloud_array, num_samples


def visualize_pointcloud_matplotlib(pointcloud, ax=None, title="Point Cloud", color_by_coords=True):
    """Visualize point cloud using matplotlib 3D."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
    
    # Extract coordinates
    if pointcloud.shape[1] >= 3:
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
    else:
        raise ValueError(f"Point cloud must have at least 3 dimensions, got {pointcloud.shape[1]}")
    
    # Color by coordinates if no RGB info
    if pointcloud.shape[1] == 3 and color_by_coords:
        # Normalize coordinates to [0, 1] for coloring
        min_coords = pointcloud.min(axis=0)
        max_coords = pointcloud.max(axis=0)
        normalized = (pointcloud - min_coords) / (max_coords - min_coords + 1e-8)
        colors = normalized
    elif pointcloud.shape[1] >= 6:
        # Use RGB colors if available (assuming channels 3-6 are RGB)
        colors = pointcloud[:, 3:6] / 255.0 if pointcloud[:, 3:6].max() > 1.0 else pointcloud[:, 3:6]
    else:
        colors = 'cyan'
    
    # Plot
    ax.scatter(x, y, z, c=colors, s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig if fig else ax.figure


def visualize_pointcloud_plotly(pointcloud, title="Point Cloud", color_by_coords=True):
    """Visualize point cloud using plotly (interactive)."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive point cloud visualization")
    
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]
    
    # Color by coordinates if no RGB info
    if pointcloud.shape[1] == 3 and color_by_coords:
        min_coords = pointcloud.min(axis=0)
        max_coords = pointcloud.max(axis=0)
        normalized = (pointcloud - min_coords) / (max_coords - min_coords + 1e-8)
        colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) 
                  for r, g, b in normalized]
    elif pointcloud.shape[1] >= 6:
        colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) 
                  for r, g, b in pointcloud[:, 3:6]]
    else:
        colors = 'cyan'
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig


def visualize_sample(img_array, depth_array, idx, pointcloud_array=None, use_plotly=False):
    """Visualize a single sample (RGB image, depth map, and optionally point cloud)."""
    # Get the sample
    img = img_array[idx]
    depth = depth_array[idx]
    
    # Ensure image is in correct format (H, W, C) and uint8
    if img.dtype != np.uint8:
        # Normalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Ensure depth is 2D
    if len(depth.shape) > 2:
        depth = depth.squeeze()
    
    # Determine number of subplots
    n_plots = 3 if pointcloud_array is not None else 2
    
    # Create figure
    if pointcloud_array is not None and use_plotly and PLOTLY_AVAILABLE:
        # Use plotly for interactive point cloud
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        
        # Display RGB image
        ax1.imshow(img)
        ax1.set_title(f'RGB Image (Sample {idx})', fontsize=12)
        ax1.axis('off')
        
        # Display depth map
        depth_im = ax2.imshow(depth, cmap='viridis')
        ax2.set_title(f'Depth Map (Sample {idx})', fontsize=12)
        ax2.axis('off')
        plt.colorbar(depth_im, ax=ax2, label='Depth')
        
        # Add depth statistics
        depth_min, depth_max = depth.min(), depth.max()
        depth_mean = depth.mean()
        info_text = f'Min: {depth_min:.3f}\nMax: {depth_max:.3f}\nMean: {depth_mean:.3f}'
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Show point cloud in separate plotly window
        pointcloud = pointcloud_array[idx]
        if len(pointcloud.shape) > 2:
            pointcloud = pointcloud.squeeze()
        pc_fig = visualize_pointcloud_plotly(pointcloud, title=f'Point Cloud (Sample {idx})')
        pc_fig.show()
        
        return fig
    else:
        # Use matplotlib for all
        if pointcloud_array is not None:
            fig = plt.figure(figsize=(18, 6))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax1, ax2 = axes
        
        # Display RGB image
        ax1.imshow(img)
        ax1.set_title(f'RGB Image (Sample {idx})', fontsize=12)
        ax1.axis('off')
        
        # Display depth map
        depth_im = ax2.imshow(depth, cmap='viridis')
        ax2.set_title(f'Depth Map (Sample {idx})', fontsize=12)
        ax2.axis('off')
        plt.colorbar(depth_im, ax=ax2, label='Depth')
        
        # Add depth statistics
        depth_min, depth_max = depth.min(), depth.max()
        depth_mean = depth.mean()
        info_text = f'Min: {depth_min:.3f}\nMax: {depth_max:.3f}\nMean: {depth_mean:.3f}'
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Display point cloud if available
        if pointcloud_array is not None:
            pointcloud = pointcloud_array[idx]
            if len(pointcloud.shape) > 2:
                pointcloud = pointcloud.squeeze()
            visualize_pointcloud_matplotlib(pointcloud, ax=ax3, title=f'Point Cloud (Sample {idx})')
        
        plt.tight_layout()
        return fig


def interactive_visualizer(img_array, depth_array, num_samples, start_idx=0, pointcloud_array=None, use_plotly=False):
    """Create an interactive visualizer with slider to navigate through samples."""
    n_plots = 3 if pointcloud_array is not None else 2
    fig = plt.figure(figsize=(6*n_plots, 6))
    plt.subplots_adjust(bottom=0.2)
    
    if pointcloud_array is not None:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')
        axes = [ax1, ax2, ax3]
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]
    
    # Initial sample
    idx = start_idx
    img = img_array[idx]
    depth = depth_array[idx]
    
    # Ensure correct format
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if len(depth.shape) > 2:
        depth = depth.squeeze()
    
    # Display RGB image
    img_im = axes[0].imshow(img)
    axes[0].set_title(f'RGB Image (Sample {idx}/{num_samples-1})', fontsize=12)
    axes[0].axis('off')
    
    # Display depth map
    depth_im = axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title(f'Depth Map (Sample {idx}/{num_samples-1})', fontsize=12)
    axes[1].axis('off')
    
    # Add colorbar for depth
    cbar = plt.colorbar(depth_im, ax=axes[1], label='Depth')
    
    # Add depth info text
    info_text = axes[1].text(0.02, 0.98, '', transform=axes[1].transAxes,
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Initialize point cloud plot if available
    pc_scatter = None
    if pointcloud_array is not None:
        pointcloud = pointcloud_array[idx]
        if len(pointcloud.shape) > 2:
            pointcloud = pointcloud.squeeze()
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        if pointcloud.shape[1] == 3:
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized = (pointcloud - min_coords) / (max_coords - min_coords + 1e-8)
            colors = normalized
        elif pointcloud.shape[1] >= 6:
            colors = pointcloud[:, 3:6] / 255.0 if pointcloud[:, 3:6].max() > 1.0 else pointcloud[:, 3:6]
        else:
            colors = 'cyan'
        pc_scatter = axes[2].scatter(x, y, z, c=colors, s=1, alpha=0.6)
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].set_zlabel('Z')
        axes[2].set_title(f'Point Cloud (Sample {idx}/{num_samples-1})', fontsize=12)
    
    def update_info():
        depth_min, depth_max = depth_array[idx].min(), depth_array[idx].max()
        depth_mean = depth_array[idx].mean()
        info_text.set_text(f'Min: {depth_min:.3f}\nMax: {depth_max:.3f}\nMean: {depth_mean:.3f}')
    
    update_info()
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Sample', 0, num_samples - 1, 
                    valinit=idx, valfmt='%d', valstep=1)
    
    def update(val):
        nonlocal idx, img, depth
        idx = int(slider.val)
        
        # Load new sample
        img = img_array[idx]
        depth = depth_array[idx]
        
        # Ensure correct format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        if len(depth.shape) > 2:
            depth = depth.squeeze()
        
        # Update images
        img_im.set_data(img)
        depth_im.set_data(depth)
        
        # Update titles
        axes[0].set_title(f'RGB Image (Sample {idx}/{num_samples-1})', fontsize=12)
        axes[1].set_title(f'Depth Map (Sample {idx}/{num_samples-1})', fontsize=12)
        
        # Update depth info
        depth_min, depth_max = depth.min(), depth.max()
        depth_mean = depth.mean()
        info_text.set_text(f'Min: {depth_min:.3f}\nMax: {depth_max:.3f}\nMean: {depth_mean:.3f}')
        
        # Update colorbar limits
        depth_im.set_clim(depth.min(), depth.max())
        
        # Update point cloud if available
        if pointcloud_array is not None:
            pointcloud = pointcloud_array[idx]
            if len(pointcloud.shape) > 2:
                pointcloud = pointcloud.squeeze()
            x = pointcloud[:, 0]
            y = pointcloud[:, 1]
            z = pointcloud[:, 2]
            if pointcloud.shape[1] == 3:
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized = (pointcloud - min_coords) / (max_coords - min_coords + 1e-8)
                colors = normalized
            elif pointcloud.shape[1] >= 6:
                colors = pointcloud[:, 3:6] / 255.0 if pointcloud[:, 3:6].max() > 1.0 else pointcloud[:, 3:6]
            else:
                colors = 'cyan'
            
            # Clear and redraw point cloud
            axes[2].clear()
            axes[2].scatter(x, y, z, c=colors, s=1, alpha=0.6)
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            axes[2].set_zlabel('Z')
            axes[2].set_title(f'Point Cloud (Sample {idx}/{num_samples-1})', fontsize=12)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.show()


def batch_visualize(img_array, depth_array, num_samples, num_samples_to_show=10, 
                    start_idx=0, save_path=None, pointcloud_array=None):
    """Visualize multiple samples in a grid."""
    num_samples_to_show = min(num_samples_to_show, num_samples - start_idx)
    
    # Calculate grid size
    cols = 3 if pointcloud_array is not None else 2
    n_per_sample = 3 if pointcloud_array is not None else 2
    rows = num_samples_to_show
    
    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    axes_flat = []
    
    for i in range(num_samples_to_show):
        idx = start_idx + i
        
        # Get samples
        img = img_array[idx]
        depth = depth_array[idx]
        
        # Ensure correct format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        if len(depth.shape) > 2:
            depth = depth.squeeze()
        
        # RGB
        ax1 = fig.add_subplot(rows, cols, i * cols + 1)
        ax1.imshow(img)
        ax1.set_title(f'RGB {idx}', fontsize=10)
        ax1.axis('off')
        
        # Depth
        ax2 = fig.add_subplot(rows, cols, i * cols + 2)
        ax2.imshow(depth, cmap='viridis')
        ax2.set_title(f'Depth {idx}', fontsize=10)
        ax2.axis('off')
        
        # Point cloud
        if pointcloud_array is not None:
            pointcloud = pointcloud_array[idx]
            if len(pointcloud.shape) > 2:
                pointcloud = pointcloud.squeeze()
            ax3 = fig.add_subplot(rows, cols, i * cols + 3, projection='3d')
            x = pointcloud[:, 0]
            y = pointcloud[:, 1]
            z = pointcloud[:, 2]
            if pointcloud.shape[1] == 3:
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized = (pointcloud - min_coords) / (max_coords - min_coords + 1e-8)
                colors = normalized
            elif pointcloud.shape[1] >= 6:
                colors = pointcloud[:, 3:6] / 255.0 if pointcloud[:, 3:6].max() > 1.0 else pointcloud[:, 3:6]
            else:
                colors = 'cyan'
            ax3.scatter(x, y, z, c=colors, s=0.5, alpha=0.6)
            ax3.set_title(f'PC {idx}', fontsize=10)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize RGB images, depth maps, and point clouds from zarr files')
    parser.add_argument('zarr_path', type=str, 
                       help='Path to the zarr file (e.g., 3D-Diffusion-Policy/data/metaworld_basketball_expert.zarr)')
    parser.add_argument('--mode', type=str, choices=['single', 'interactive', 'batch', 'pointcloud'], 
                       default='interactive',
                       help='Visualization mode: single (one sample), interactive (with slider), batch (grid), or pointcloud (point cloud only)')
    parser.add_argument('--idx', type=int, default=0,
                       help='Starting index for visualization (default: 0)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to show in batch mode (default: 10)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the visualization (only for single/batch mode)')
    parser.add_argument('--no-pointcloud', action='store_true',
                       help='Skip loading point clouds even if available')
    parser.add_argument('--use-plotly', action='store_true',
                       help='Use plotly for interactive 3D point cloud visualization (requires plotly)')
    
    args = parser.parse_args()
    
    # Expand user path and resolve
    zarr_path = os.path.expanduser(args.zarr_path)
    zarr_path = os.path.abspath(zarr_path)
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr file not found at: {zarr_path}")
    
    # Load data
    load_pc = not args.no_pointcloud
    img_array, depth_array, pointcloud_array, num_samples = load_zarr_data(zarr_path, load_pointcloud=load_pc)
    
    # Check index validity
    if args.idx >= num_samples:
        print(f"Warning: Index {args.idx} is out of range. Using index 0 instead.")
        args.idx = 0
    
    # Visualize based on mode
    if args.mode == 'pointcloud':
        # Point cloud only mode
        if pointcloud_array is None:
            raise ValueError("No point cloud data available in zarr file")
        if args.use_plotly and PLOTLY_AVAILABLE:
            pointcloud = pointcloud_array[args.idx]
            if len(pointcloud.shape) > 2:
                pointcloud = pointcloud.squeeze()
            fig = visualize_pointcloud_plotly(pointcloud, title=f'Point Cloud (Sample {args.idx})')
            if args.save:
                fig.write_html(args.save)
                print(f"Saved visualization to {args.save}")
            else:
                fig.show()
        else:
            fig = visualize_pointcloud_matplotlib(pointcloud_array[args.idx], 
                                                  title=f'Point Cloud (Sample {args.idx})')
            if args.save:
                fig.savefig(args.save, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {args.save}")
            else:
                plt.show()
    
    elif args.mode == 'single':
        fig = visualize_sample(img_array, depth_array, args.idx, 
                              pointcloud_array=pointcloud_array, use_plotly=args.use_plotly)
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {args.save}")
        else:
            plt.show()
    
    elif args.mode == 'interactive':
        interactive_visualizer(img_array, depth_array, num_samples, start_idx=args.idx,
                              pointcloud_array=pointcloud_array, use_plotly=args.use_plotly)
    
    elif args.mode == 'batch':
        batch_visualize(img_array, depth_array, num_samples, 
                       num_samples_to_show=args.num_samples,
                       start_idx=args.idx, save_path=args.save, pointcloud_array=pointcloud_array)


if __name__ == '__main__':
    main()

