import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
cache_dir = Path(os.environ.get("CACHE_DIR", ROOT_DIR / "cache"))
os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "vggt"))

import numpy as np
import cv2
from src.reconstruction.vggt_reconstructor import VGGTReconstructor
from src.mapping.occupancy_grid import OccupancyGridGenerator, GridMapConfig
from src.planning.path_planner import AStarPlanner


def main():
    print("=== Testing Occupancy Grid Generation & Path Planning ===")
    room_dir = str(ROOT_DIR / "vggt" / "examples" / "room" / "images")
    
    # 1. Reconstruct
    reconstructor = VGGTReconstructor()
    reconstruction = reconstructor.reconstruct_from_images(room_dir)
    
    points = reconstruction["pcd_points"]
    extrinsics = reconstruction["extrinsic"] # (S, 3, 4)
    # Camera centers in world frame: - R^T @ t
    cam_centers = []
    for ext in extrinsics:
        R = ext[:, :3]
        t = ext[:, 3]
        center = -R.T @ t
        cam_centers.append(center)
    cam_centers = np.array(cam_centers)

    # 2. Generate 2D Occupancy Grid
    map_gen = OccupancyGridGenerator(GridMapConfig(resolution=0.05, robot_radius=0.15))
    grid_info = map_gen.generate_grid_map(points, camera_centers=cam_centers)

    inflated_grid = grid_info["inflated_grid"]
    occ_grid = grid_info["occupancy_grid"]
    origin = grid_info["origin"]
    res = grid_info["resolution"]
    R_align = grid_info["rotation_matrix"]
    h, w = inflated_grid.shape

    print(f"Occupancy Grid Size: {w}x{h} cells (resolution: {res}m)")
    print(f"Obstacle cells: {np.sum(occ_grid)} / {occ_grid.size} ({np.mean(occ_grid)*100:.1f}%)")
    print(f"Inflated cells: {np.sum(inflated_grid)} / {inflated_grid.size} ({np.mean(inflated_grid)*100:.1f}%)")

    # 3. Path Planning from Camera 0 to an arbitrary point
    cam0_aligned = (R_align @ cam_centers[0].reshape(3, 1)).flatten()
    start_c = int((cam0_aligned[0] - origin[0]) / res)
    start_r = int((cam0_aligned[1] - origin[1]) / res)
    start_cell = (start_r, start_c)

    # Goal: pick a free cell on the opposite side
    free_cells = np.argwhere(~inflated_grid)
    if len(free_cells) > 0:
        # Pick cell furthest from start
        distances = np.linalg.norm(free_cells - np.array(start_cell), axis=1)
        goal_cell = tuple(free_cells[np.argmax(distances)])
    else:
        goal_cell = (h // 2, w // 2)

    planner = AStarPlanner()
    path = planner.plan_path(inflated_grid, start_cell, goal_cell)

    if path:
        print(f"A* Path found! Length: {len(path)} waypoints.")
        smoothed_path = planner.smooth_path(inflated_grid, path)
        print(f"Smoothed path length: {len(smoothed_path)} waypoints.")
    else:
        print("No path found.")
        smoothed_path = []

    # 4. Render 2D map with path to an image
    # RGB image: White = free, Black = obstacle, Gray = inflated, Red = path, Green = start, Blue = goal
    vis_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    vis_img[inflated_grid] = [180, 180, 180]
    vis_img[occ_grid] = [30, 30, 30]

    # Draw path
    if path:
        for r, c in path:
            if 0 <= r < h and 0 <= c < w:
                vis_img[r, c] = [0, 0, 255] # Red

        for r, c in smoothed_path:
            cv2.circle(vis_img, (c, r), 2, (255, 0, 0), -1) # Blue waypoints

    # Start and Goal markers
    cv2.circle(vis_img, (start_cell[1], start_cell[0]), 4, (0, 255, 0), -1) # Green
    cv2.circle(vis_img, (goal_cell[1], goal_cell[0]), 4, (0, 165, 255), -1) # Orange

    # Flip vertically for standard top-down image orientation
    vis_img = np.flipud(vis_img)
    out_map_path = str(ROOT_DIR / "data" / "occupancy_map_with_path.png")
    cv2.imwrite(out_map_path, vis_img)
    print(f"Map with planned path saved to: {out_map_path}")
    print("=== Mapping & Planning Test Finished Successfully ===")


if __name__ == "__main__":
    main()
