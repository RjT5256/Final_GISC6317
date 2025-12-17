# %% Importsz
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import laspy
import os

#%% Step 1: Load Point Cloud Data
def load_point_cloud(file_path):
    """
    Load point cloud data from LAS/LAZ or PLY files
    """
    if file_path.endswith('.las') or file_path.endswith('.laz'):
        # Load LAS/LAZ file
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif file_path.endswith('.ply'):
        # Load PLY file directly with Open3D
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd

# %% User Input and Execution 🚀
while True:
    file_path_input = input("Enter the full path to the LAS/LAZ/PLY file: ")

    if os.path.exists(file_path_input):
        try:
            pcd = load_point_cloud(file_path_input)
            break
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Please try a different file path.")
    else:
        print(f"Error: File not found at path: {file_path_input}")
        print("Please ensure the path is correct and the file exists.")

o3d.visualization.draw_geometries([pcd])

#%% Step 2: Preprocess point clouds
def preprocess_point_cloud(pcd, voxel_size=.05,
remove_outliers=True):
    """
    Preprocess point cloud: downsamplnig and outlier removal
    """
    # Downsample using voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    #Estimate normals for the downampled point cloud
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Remove outliers if requested
    if remove_outliers:
        #statistical outlier removeal
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print(f"Preprocessed cloud has {len(pcd_down.points)} points")
    return pcd_down

#Test
pcd_processed = preprocess_point_cloud(pcd, voxel_size=.01)
o3d.visualization.draw_geometries([pcd_processed])
    
# %% Step 3: Load the second temporality 🚀

while True:
    file_path_t1 = input("Enter the full path to the SECOND (T1) LAS/LAZ/PLY file: ")

    if os.path.exists(file_path_t1):
        try:
            pcd_t1 = load_point_cloud(file_path_t1)
            break
        except Exception as e:
            print(f"Error loading T1 file: {e}")
            print("Please try a different file path for the T1 data.")
    else:
        print(f"Error: File not found at path: {file_path_t1}")
        print("Please ensure the T1 path is correct and the file exists.")

pcd_processed_t1 = preprocess_point_cloud(pcd_t1, voxel_size=.1)
pcd_processed.paint_uniform_color([0,1,0])
pcd_processed_t1.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([pcd_processed, pcd_processed_t1])

#%% Step 4: Determining Geo-Referencing errors

def register_point_clouds(source,target,voxel_size=.05,max_iter=100):
    """
    Register source point cloud to target using point-to-plane ICP
    """
    
    #Initialize transformation with identity matrix(4)
    init_transformation = np.identity(4)
    
    #set convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iter
    )
    
    #Point-to-place ICP registration
    result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size*2, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    #Apply transformation to source
    source_transformed = source.transform(result.transformation)
    
    print(f"Registration finished with fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
    return source_transformed, result.transformation

#Let us test our functions
target_pcd = pcd_processed

source_pcd = preprocess_point_cloud(load_point_cloud(file_path_input), voxel_size=.1)
source_aligned, transformation = register_point_clouds(source_pcd, target_pcd, voxel_size=.1)

target_pcd.paint_uniform_color([1,0,0])
source_pcd.paint_uniform_color([0,0,1])
source_aligned.paint_uniform_color([0,1,0])

o3d.visualization.draw_geometries([source_pcd, source_aligned, target_pcd])

#%% Step 5: Compute Cloud-to-Cloud Distance
def compute_cloud_distances(source, target):
    """
    Compute point-to-point distances between source and target clouds
    """
    
    #Convert target points to numpy array for KDTree
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    
    #Build KDTree from target points
    tree = KDTree(target_points)
    
    #Query the tree for nearest neighbor distances
    distances, _ = tree.query(source_points)
    
    print(f"Computed distances for {len(source_points)} points")
    return distances

#renaming conventions
source_aligned = pcd_processed
target_pcd = pcd_processed_t1

# distances = compute_cloud_distances(pcd_processed, pcd_processed_t1)
distances = compute_cloud_distances(source_aligned, target_pcd)

#%% Step 6 Statistical Analysis of Changes
def analyze_changes(distances, threshold=.1):
    """
    Analyze distances to identify significant changes
    """
    # Identify points with distance greater than trheshold
    change_indices = np.where(distances > threshold)[0]
    change_distances = distances[change_indices]
    
    # Calcualte statistics
    if len(change_distances) > 0:
        mean_change = np.mean(change_distances)
        max_change = np.max(change_distances)
        total_volume_change = len(change_indices) / len(distances) # Approx. as percentage of points
        
        print(f"Detexted {len(change_indices)} points with significant change")
        print(f"Mean change: {mean_change:.3f}m, Max change: {max_change:.3f}m")
        print(f"Approximate volume change: {total_volume_change*100:.2f}%")
        
        return change_indices, {
            "mean_change": mean_change,
            "max_change": max_change,
            "volume_change_percentage": total_volume_change*100
            }

# Test
change_indices, stats = analyze_changes(distances, threshold=.1)

#%% Step 7: Create Distance Heatmap and colorize point cloud 
def create_distance_heatmap(source, distances):
    """
    Visualize the entire point cloud as a heatmap based on distance values
    """
    
    #Create a copy of the source cloud
    heatmap_pcd = o3d.geometry.PointCloud()
    heatmap_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    
    #Normalize distances dor visualization
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    #Create a colormap (blue=close, red=far)
    if max_dist > min_dist:
        normalized_dists = (distances - min_dist) / (max_dist - min_dist)
    else:
        normalized_distslized_dists = np.ones_like(distances) * .5
        
    #Create color array using a gradient from blue to red
    colors = np.zeros((len(distances), 3))
    colors[:, 0] = normalized_dists #red channel increases with distance
    colors[:, 2] = 1 - normalized_dists # blue channel decreases with distance
        
    #Add green component for a more dynamic color change
    colors[:, 1] = np.where(normalized_dists < .5,
                                normalized_dists * 2,
                                (1- normalized_dists) * 2)
        
    heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
        
    print(f"Heatmap color scale: Blue = {min_dist:.3f}m, Red = {max_dist:.3f}m")
        
    return heatmap_pcd

heatmap_pcd = create_distance_heatmap(source_aligned, distances)
o3d.visualization.draw_geometries([heatmap_pcd])

#%% Step 8: Object-based Change-detection (Advanced)
def detect_missing_regions(source, target, distances,
                           distance_threshold=.1, region_size_threshold=10):
    """
    Detect regions in source that have no correspondence in target using distance thresholding and region growing
    """
    source_points = np.asarray(source.points)
    
    # Find points that are far from any points in the target potentially missing)
    missing_indices = np.where(distances > distance_threshold)[0]
    
    if len(missing_indices) == 0:
        print("No significant differences detected")
        return [],[],[]
    
    #Create a KDTree of the source points
    source_tree = KDTree(source.points)
    
    # Initiatlize variables for region growing
    all_regions = []
    processed = np.zeros(len(source_points), dtype=bool)
    
    # Process each unprocessed missing point
    for idx in missing_indices:
        if processed[idx]:
            continue
        
        #Start a new region with this point
        current_region = [idx]
        processed[idx] = True
        
        # Grow the region
        i = 0 
        while i < len(current_region):
            #Get current point in the growing region
            current_idx = current_region[i]
            
            #Find neighbors within the 3D neighborhood (using KDTree)
            neighbors_dist, neighbors_idx = source_tree.query(source_points[current_idx].reshape(1, -1), k=20
                                                              )
            #Add unprocessed neighbors that are also missing
            for neighbor_idx in neighbors_idx[0][1:]:
                if not processed[neighbor_idx] and neighbor_idx in missing_indices:
                    current_region.append(neighbor_idx)
                    processed[neighbor_idx] = True
            i+=1
            
        # Store region if it's large enough (to filter out noise)
        if len(current_region) >= region_size_threshold:
            all_regions.append(current_region)
            
    #Flatten all regions into a single list of indices
    all_missing_indices = []
    region_labels = np.zeros(len(source_points), dtype=int)
    
    for region_idx, region in enumerate(all_regions, 1):
        all_missing_indices.extend(region)
        #Label each point with its region number
        for point_idx in region:
            region_labels[point_idx] = region_idx
            
    print(f"Detected {len(all_regions)} missing regions with total {len(all_missing_indices)} pints")
    
    #Return missing regions, all missing indices, and region lables
    return all_regions, np.array(all_missing_indices), region_labels

# Test
regions, missing_indices, region_labels = detect_missing_regions(source_aligned, target_pcd, distances,
                                                               distance_threshold=.15)

#%% Step 9: Visualize Changes with Color-coded Regions
def visualize_colored_changes(source, target, distances, distance_threshold=.1, region_size_threshold=10):
    """
    Visualize changes by coloring the entire source point cloud based on detected changes
    """
    
    #First detect regions
    regions, missing_indices, region_labels = detect_missing_regions(
        source, target, distances, distance_threshold, region_size_threshold)
    
    if len(regions) == 0:
        print("No significant changes to visualize")
        # Just show the two point clouds in different colors
        source.paint_uniform_color([.7,.7,.7])
        target.paint_uniform_color([.5,.5,.8])
        o3d.visualization.draw_geomtries([source, target])
        return None, None
          
    # Create a colored copy of the source point cloud
    colored_source = o3d.geometry.PointCloud()
    colored_source.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    
    #Initialize colors for all points (default to gray for unchanged points)
    num_points = len(source.points)
    colors = np.ones((num_points, 3)) * .7 #Gray for unchanged points
    
    # Generate unique colors for each region
    num_regions = len(regions)
    region_colors = np.zeros((num_regions + 1, 3))
    
    # Create a colormap for the regions
    # Using a perceptually distinct color palette
    base_colors = [
        [1, 0 , 0], #Red
        [1, 0 , 0], #Blue
        [1, 0 , 0], #Green
        [1, 0 , 0], #Yellow
        [1, 0 , 0], #Magenta
        [1, 0 , 0], #Cyan
        [1, 0 , 0], #Orange
        [1, 0 , 0], #Purple
        [1, 0 , 0], #Dark Green
        [1, 0 , 0]  #Olive
        ]
    
    #Assign colors to regions, cycling through the base colors
    for j in range(1, num_regions + 1):
            region_colors[j] = base_colors[(j - 1) % len(base_colors)]

    # 1. Iterate through all points to apply region colors
    for i in range(num_points):
        # region_labels[i] holds the index of the region (0 for unchanged, 1+ for regions)
        if region_labels[i] > 0:
            # Assign the unique color of the detected region
            colors[i] = region_colors[region_labels[i]]
    
    # Apply colors to the point cloud
    colored_source.colors = o3d.utility.Vector3dVector(colors)
    
    #Color the target point cloud for contrast
    target.paint_uniform_color([.5,.5,.8]) # Light blue
    
    #Visualize the point clouds
    print(f"Visualizing source with {len(regions)} colored missing regions")
    o3d.visualization.draw_geometries([colored_source, target])
    
    #Create a seperate point cloud just for the missing points
    missing_pcd = o3d.geometry.PointCloud()
    source_points = np.asarray(source.points)
    missing_points = source_points[missing_indices]
    missing_pcd.points = o3d.utility.Vector3dVector(missing_points)
    
    return colored_source, missing_pcd

#Test
colored_source, missing_pcd = visualize_colored_changes(source_aligned, target_pcd, distances, distance_threshold=.15)
    
#%% Step 10: Upsampling predictions to the full point cloud
def transfer_colors_to_original(original_pcd, colored_downsampled_pcd):

    #Create a copy of the original point cloud to add colors to
    colored_original = o3d.geometry.PointCloud()
    colored_original.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))

    #Get numpy arrays of points
    original_points = np.asarray(original_pcd.points)
    downsampled_points = np.asarray(colored_downsampled_pcd.points)
    downsampled_colors = np.asarray(colored_downsampled_pcd.colors)

    # Build KDTree(downsampled_points)
    tree = KDTree(downsampled_points)

    # For each point in the original cloud, find the nearest neighbor in the downsampled cloud
    _, indices = tree.query(original_points)

    # Trasnfer colors based on nearest neighbor relationship
    original_colors = downsampled_colors[indices]
    colored_original.colors = o3d.utility.Vector3dVector(original_colors)

    print(f"Transferred colors from downsampled cloud ({len(downsampled_points)} points) to original cloud ({len(original_points)} points)")

    return colored_original

#Test
original_colored = transfer_colors_to_original(pcd, colored_source)
original_colored.estimate_normals()
o3d.visualization.draw_geometries([original_colored])

#%% Step 11: Preparing and exportng the results
def save_results(colored_source, missing_pcd, heatmap_pcd, regions, stats, output_dir="./output"):
    """
    Save all visualizations and analysis results
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the colored point cloud
    if colored_source is not None:
        o3d.io.write_point_cloud(f"{output_dir}/colored_source.ply", original_colored)
        print(f"Saved colored source point cloud to {output_dir}/colored_source.ply")

    # Save the missing regions point cloud
    if missing_pcd is not None and len(missing_pcd.points) > 0:
        o3d.io.write_point_cloud(f"{output_dir}/missing_regions.ply" , missing_pcd)
        print(f"Saved missing regions point cloud to {output_dir}/missing_regions.ply")

    # Save the heatmap point cloud
    if heatmap_pcd is not None:
        o3d.io.write_point_cloud(f"{output_dir}/distance_heatmap.ply" , heatmap_pcd)
        print(f"Saved distance heatmap to {output_dir}/distance_heatmap.ply")

    # Save region statistics
    if regions:
        with open(f"{output_dir}/region_stats.txt", "w") as f:
            f.write(f"Total number of detected regions: {len(regions)}\n\n")
            for i, region in enumerate(regions):
                f.write(f"Region {i+1}: {len(region)} points\n")
        print(f"Saved region statistics to {output_dir}/region_stats.txt")

    # Save overall statistics
    with open(f"{output_dir}/change_stats.txt" , "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved overall statistics to {output_dir}/ change_stats.txt")

#Test
save_results(colored_source, missing_pcd, heatmap_pcd, regions, stats)
# %%
