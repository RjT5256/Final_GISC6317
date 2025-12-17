# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 08:54:11 2025

@author: rober
"""

# %% Imports
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import laspy
import os
import sys

# %% The Application Logic
class PointCloudApp:
    def __init__(self):
        # Establishing step by step procedure of defined methods from previous base file for command use
        self.pcd_source = None
        self.pcd_target = None
        self.pcd_source_proc = None
        self.pcd_target_proc = None
        self.distances = None
        self.regions = None
        self.colored_source = None
        self.missing_pcd = None
        self.stats = None

    # Loading the methods from the original file
    def load_point_cloud(self, prompt_text="Enter path to LAS/LAZ/PLY file: "):
        while True:
            file_path = input(prompt_text).strip().strip('"').strip("'")
            
            if os.path.exists(file_path):
                try:
                    print(f"Loading {file_path}...")
                    if file_path.lower().endswith(('.las', '.laz')):
                        las = laspy.read(file_path)
                        points = np.vstack((las.x, las.y, las.z)).transpose()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                    elif file_path.lower().endswith('.ply'):
                        pcd = o3d.io.read_point_cloud(file_path)
                    else:
                        print("Unsupported format. Please use .las, .laz, or .ply")
                        continue
                    
                    print(f"✅ Loaded {len(pcd.points)} points.")
                    return pcd
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
            else:
                print(f"❌ File not found: {file_path}")
                return None # Allow breaking out if needed or loop

    def preprocess_point_cloud(self, pcd, voxel_size=0.05, remove_outliers=True):
        print("Preprocessing (Downsampling + Normals + Outlier Removal)...")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        if remove_outliers:
            pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd_down

    def register_point_clouds(self, source, target, voxel_size=0.05, max_iter=100):
        print("Aligning Source (T0) to Target (T1)...")
        init_transformation = np.identity(4)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter)
        
        result = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size*2, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria)
        
        source_transformed = source.transform(result.transformation)
        print(f"✅ Registration Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
        return source_transformed

    def compute_cloud_distances(self, source, target):
        print("Computing cloud-to-cloud distances...")
        target_points = np.asarray(target.points)
        source_points = np.asarray(source.points)
        tree = KDTree(target_points)
        distances, _ = tree.query(source_points)
        return distances

    def analyze_changes(self, distances, threshold=0.1):
        change_indices = np.where(distances > threshold)[0]
        change_distances = distances[change_indices]
        
        stats = {}
        if len(change_distances) > 0:
            stats = {
                "mean_change": np.mean(change_distances),
                "max_change": np.max(change_distances),
                "volume_change_percentage": (len(change_indices) / len(distances)) * 100
            }
            print(f"📊 Stats: {len(change_indices)} changed points ({stats['volume_change_percentage']:.2f}%)")
        else:
            print("No significant changes detected.")
        return change_indices, stats

    def create_distance_heatmap(self, source, distances):
        print("Generating Heatmap...")
        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.points))
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist > min_dist:
            normalized_dists = (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_dists = np.zeros_like(distances)
            
        colors = np.zeros((len(distances), 3))
        colors[:, 0] = normalized_dists  # Red increases with distance
        colors[:, 2] = 1 - normalized_dists # Blue decreases with distance
        colors[:, 1] = np.where(normalized_dists < 0.5, normalized_dists * 2, (1 - normalized_dists) * 2)
        
        heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Heatmap scale: Blue (0m) -> Red ({max_dist:.3f}m)")
        return heatmap_pcd

    def detect_missing_regions(self, source, distances, distance_threshold=0.1, region_size_threshold=10):
        print("Detecting Missing Object Regions...")
        source_points = np.asarray(source.points)
        missing_indices = np.where(distances > distance_threshold)[0]
        
        if len(missing_indices) == 0:
            return [], [], np.zeros(len(source_points), dtype=int)

        source_tree = KDTree(source.points)
        all_regions = []
        processed = np.zeros(len(source_points), dtype=bool)
        
        # Region growing logic
        for idx in missing_indices:
            if processed[idx]: continue
            
            current_region = [idx]
            processed[idx] = True
            i = 0
            while i < len(current_region):
                current_idx = current_region[i]
                # Query neighbors
                neighbors_dist, neighbors_idx = source_tree.query(source_points[current_idx].reshape(1, -1), k=20)
                for neighbor_idx in neighbors_idx[0][1:]:
                    if not processed[neighbor_idx] and neighbor_idx in missing_indices:
                        current_region.append(neighbor_idx)
                        processed[neighbor_idx] = True
                i += 1
            
            if len(current_region) >= region_size_threshold:
                all_regions.append(current_region)

        # Label generation
        region_labels = np.zeros(len(source_points), dtype=int)
        all_missing_indices = []
        for region_idx, region in enumerate(all_regions, 1):
            all_missing_indices.extend(region)
            for point_idx in region:
                region_labels[point_idx] = region_idx
                
        print(f"✅ Detected {len(all_regions)} distinct changed regions.")
        return all_regions, np.array(all_missing_indices), region_labels

    def visualize_colored_changes(self, source, region_labels, regions):
        colored_source = o3d.geometry.PointCloud()
        colored_source.points = o3d.utility.Vector3dVector(np.asarray(source.points))
        
        num_points = len(source.points)
        colors = np.ones((num_points, 3)) * 0.7 # Default Gray
        
        # Base color palette from original file being used
        base_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
            [0, 1, 1], [1, 0, 1], [1, 0.5, 0], [0.5, 0, 0.5]
        ]
        
        for i in range(num_points):
            if region_labels[i] > 0:
                # Cycling base colors...
                color_idx = (region_labels[i] - 1) % len(base_colors)
                colors[i] = base_colors[color_idx]

        colored_source.colors = o3d.utility.Vector3dVector(colors)
        return colored_source

    def transfer_colors_to_original(self, original_pcd, colored_downsampled):
        print("Upsampling colors to original high-res cloud...")
        colored_original = o3d.geometry.PointCloud()
        colored_original.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))
        
        tree = KDTree(np.asarray(colored_downsampled.points))
        _, indices = tree.query(np.asarray(original_pcd.points))
        
        original_colors = np.asarray(colored_downsampled.colors)[indices]
        colored_original.colors = o3d.utility.Vector3dVector(original_colors)
        return colored_original

    def save_results(self, output_dir="./output"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        print(f"Saving to {output_dir}...")
        if self.colored_source:
            # Unsampling point cloud
            if self.pcd_source:
                final_cloud = self.transfer_colors_to_original(self.pcd_source, self.colored_source)
                o3d.io.write_point_cloud(f"{output_dir}/colored_source_fullres.ply", final_cloud)
            else:
                o3d.io.write_point_cloud(f"{output_dir}/colored_source_lowres.ply", self.colored_source)

        if self.missing_pcd:
            o3d.io.write_point_cloud(f"{output_dir}/missing_regions.ply", self.missing_pcd)
            
        if self.stats:
            with open(f"{output_dir}/change_stats.txt", "w") as f:
                for k, v in self.stats.items(): f.write(f"{k}: {v}\n")
        print("✅ Save complete.")

    # Loop through the menus
    def run(self):
        print("\n=== Point Cloud Change Detection Tool ===")
        while True:
            print("\nMAIN MENU:")
            print("1. Load Source (T0) and Target (T1) Files")
            print("2. Process Data (Downsample, Register, Compute Distances)")
            print("3. View Distance Heatmap")
            print("4. Detect & Visualize Changed Objects (Regions)")
            print("5. Save Results")
            print("6. Exit")
            
            choice = input("\nEnter selection (1-6): ")
            
            if choice == '1':
                print("\n--- Loading Data ---")
                self.pcd_source = self.load_point_cloud("Path to SOURCE (T0) file: ")
                if self.pcd_source:
                    self.pcd_target = self.load_point_cloud("Path to TARGET (T1) file: ")
                    # View raw files
                    if self.pcd_target:
                        print("Visualizing Raw Inputs...")
                        self.pcd_source.paint_uniform_color([0,1,0]) # Green
                        self.pcd_target.paint_uniform_color([1,0,0]) # Red
                        o3d.visualization.draw_geometries([self.pcd_source, self.pcd_target], window_name="Raw Inputs")

            elif choice == '2':
                if not self.pcd_source or not self.pcd_target:
                    print("⚠️ Please Load Data (Option 1) first.")
                    continue
                
                # Establishing the first initial steps as one large all-encompassing pass
                self.pcd_source_proc = self.preprocess_point_cloud(self.pcd_source, voxel_size=0.1)
                self.pcd_target_proc = self.preprocess_point_cloud(self.pcd_target, voxel_size=0.1)
                self.pcd_source_proc = self.register_point_clouds(self.pcd_source_proc, self.pcd_target_proc)
                self.distances = self.compute_cloud_distances(self.pcd_source_proc, self.pcd_target_proc)
                
                print("✅ Processing complete. You can now view Heatmaps or Regions.")

            elif choice == '3':
                if self.distances is None:
                    print("⚠️ Please Process Data (Option 2) first.")
                    continue
                
                heatmap = self.create_distance_heatmap(self.pcd_source_proc, self.distances)
                o3d.visualization.draw_geometries([heatmap], window_name="Distance Heatmap")

            elif choice == '4':
                if self.distances is None:
                    print("⚠️ Please Process Data (Option 2) first.")
                    continue
                
                # Analysis
                threshold = float(input("Enter distance threshold (default 0.15): ") or 0.15)
                _, self.stats = self.analyze_changes(self.distances, threshold=threshold)
                
                # Region detection
                self.regions, missing_indices, region_labels = self.detect_missing_regions(
                    self.pcd_source_proc, self.distances, distance_threshold=threshold)
                
                if len(self.regions) > 0:
                    # Colorize
                    self.colored_source = self.visualize_colored_changes(
                        self.pcd_source_proc, region_labels, self.regions)
                    
                    # Create isolated missing points cloud
                    self.missing_pcd = o3d.geometry.PointCloud()
                    pts = np.asarray(self.pcd_source_proc.points)
                    self.missing_pcd.points = o3d.utility.Vector3dVector(pts[missing_indices])
                    self.missing_pcd.paint_uniform_color([1, 0, 0]) # Red for missing
                    
                    print("Visualizing Colored Regions...")
                    # Show colored source vs target
                    self.pcd_target_proc.paint_uniform_color([0.5, 0.5, 0.5]) # Gray target
                    o3d.visualization.draw_geometries([self.colored_source, self.pcd_target_proc], 
                                                      window_name="Detected Changes")
                else:
                    print("No significant regions found to visualize.")

            elif choice == '5':
                if self.colored_source is None:
                    print("⚠️ Nothing to save yet. Run detection (Option 4).")
                    continue
                self.save_results()

            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid selection. Try again.")

# %% Execution
if __name__ == "__main__":
    app = PointCloudApp()
    app.run()