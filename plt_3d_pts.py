import open3d as o3d
import pandas as pd
import argparse

def read_and_plot_3d_points(file_path):
    # Read the text file into a DataFrame
    try:
        df = pd.read_csv(file_path, header=None, names=['x', 'y', 'z'])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check if file contains correct columns
    if df.shape[1] != 3:
        print("Error: File must contain exactly 3 values (x,y,z) per line separated by commas.")
        return

    # Extract x, y, z coordinates
    points = df.to_numpy()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D points from a text file using Open3D")
    parser.add_argument("file_path", type=str, help="Path to the input text file")
    args = parser.parse_args()

    read_and_plot_3d_points(args.file_path)
