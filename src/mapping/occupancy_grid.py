"""
2D Occupancy Grid Generation Module
Estimates horizontal ground plane via RANSAC, delineates room interior boundaries,
and projects 3D point clouds to a 2D navigation grid (preventing paths from leaving the room).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class GridMapConfig:
    resolution: float = 0.05           # Grid cell size in meters (5 cm)
    min_obstacle_height: float = 0.12  # Min height above ground for obstacles (12 cm)
    max_obstacle_height: float = 1.80  # Max height above ground for obstacles (1.80 m)
    robot_radius: float = 0.10         # Inflation radius around obstacles (10 cm)
    floor_margin: float = 0.06         # Distance threshold around ground plane (6 cm)


class OccupancyGridGenerator:
    def __init__(self, config: Optional[GridMapConfig] = None):
        self.config = config or GridMapConfig()

    @staticmethod
    def fit_floor_plane_ransac(
        points: np.ndarray,
        distance_threshold: float = 0.04,
        max_iterations: int = 600,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits the horizontal ground plane (in OpenCV camera frame, floor is at bottom, high Y).
        Constrains the plane normal to be primarily vertical to never fit vertical walls.
        """
        num_points = points.shape[0]
        if num_points < 3:
            raise ValueError("Need at least 3 points to fit a plane.")

        # Subsample for fast and robust fitting
        step = max(1, num_points // 25000)
        pts_sub = points[::step]

        # In OpenCV coordinates, Y is downwards, so floor is at bottom (high positive Y)
        y_thresh = np.percentile(pts_sub[:, 1], 60)
        candidate_indices = np.where(pts_sub[:, 1] >= y_thresh)[0]

        if len(candidate_indices) < 20:
            candidate_indices = np.arange(len(pts_sub))

        candidate_pts = pts_sub[candidate_indices]

        best_inliers = 0
        best_plane = np.array([0.0, -1.0, 0.0, 0.0])

        for _ in range(max_iterations):
            sample_ids = np.random.choice(len(candidate_pts), size=3, replace=False)
            p1, p2, p3 = candidate_pts[sample_ids]

            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm < 1e-5:
                continue
            normal /= norm

            # MUST BE HORIZONTAL: in OpenCV camera coordinates, Y is down/up,
            # so the floor normal must have high absolute Y (|normal[1]| > 0.65).
            # This completely rejects vertical walls!
            if abs(normal[1]) < 0.65:
                continue

            d = -np.dot(normal, p1)
            dists = np.abs(np.dot(candidate_pts, normal) + d)
            inliers = np.sum(dists < distance_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.append(normal, d)

        # Orient normal UPWARDS (towards -Y, i.e., ceiling in OpenCV)
        if best_plane[1] > 0:
            best_plane = -best_plane

        normal = best_plane[:3]
        d = best_plane[3]

        # Compute inliers on full point cloud
        full_dists = np.abs(np.dot(points, normal) + d)
        inlier_mask = full_dists < distance_threshold

        return best_plane, inlier_mask

    @staticmethod
    def compute_align_to_z_rotation(normal: np.ndarray) -> np.ndarray:
        """
        Calculates rotation matrix R such that R @ normal = [0, 0, 1] (+Z upwards).
        """
        normal = normal / np.linalg.norm(normal)
        target = np.array([0.0, 0.0, 1.0])

        if np.allclose(normal, target):
            return np.eye(3)
        if np.allclose(normal, -target):
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c = np.dot(normal, target)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        r = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s**2))
        return r

    def generate_grid_map(
        self,
        points: np.ndarray,
        camera_centers: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Projects 3D point cloud to an aligned 2D occupancy grid strictly bounded by the room interior.
        """
        # 1. Fit true horizontal floor plane
        plane, inliers = self.fit_floor_plane_ransac(points)
        normal = plane[:3]
        d = plane[3]

        # 2. Align points so floor normal is along +Z (vertical)
        r_align = self.compute_align_to_z_rotation(normal)
        aligned_pts = (r_align @ points.T).T

        # In aligned frame: ground normal is (0,0,1), so ground plane equation is z + d = 0 -> z = -d
        ground_z = -d

        # 3. Separate ground points and obstacle points
        heights = aligned_pts[:, 2] - ground_z

        ground_mask = np.abs(heights) <= self.config.floor_margin
        obstacle_mask = (heights >= self.config.min_obstacle_height) & (
            heights <= self.config.max_obstacle_height
        )

        ground_pts = aligned_pts[ground_mask]
        obstacle_pts = aligned_pts[obstacle_mask]

        # 4. Rasterize to 2D Grid
        res = self.config.resolution
        min_x = np.percentile(aligned_pts[:, 0], 0.5) - 0.10
        max_x = np.percentile(aligned_pts[:, 0], 99.5) + 0.10
        min_y = np.percentile(aligned_pts[:, 1], 0.5) - 0.10
        max_y = np.percentile(aligned_pts[:, 1], 99.5) + 0.10

        grid_w = int(np.ceil((max_x - min_x) / res))
        grid_h = int(np.ceil((max_y - min_y) / res))

        # 5. Delineate room interior boundary from observed floor points
        # Only cells with observed ground points (or within 25cm of them) are inside the room!
        floor_obs = np.zeros((grid_h, grid_w), dtype=bool)
        if len(ground_pts) > 0:
            gc = np.clip(((ground_pts[:, 0] - min_x) / res).astype(int), 0, grid_w - 1)
            gr = np.clip(((ground_pts[:, 1] - min_y) / res).astype(int), 0, grid_h - 1)
            floor_obs[gr, gc] = True

        dilation_radius_cells = int(np.ceil(0.30 / res))  # 30 cm floor filling
        yk, xk = np.ogrid[
            -dilation_radius_cells : dilation_radius_cells + 1,
            -dilation_radius_cells : dilation_radius_cells + 1,
        ]
        disk = xk**2 + yk**2 <= dilation_radius_cells**2
        room_interior = ndimage.binary_dilation(floor_obs, structure=disk)

        # 6. Obstacle grid inside the room
        obs_grid = np.zeros((grid_h, grid_w), dtype=bool)
        if len(obstacle_pts) > 0:
            oc = np.clip(((obstacle_pts[:, 0] - min_x) / res).astype(int), 0, grid_w - 1)
            orow = np.clip(((obstacle_pts[:, 1] - min_y) / res).astype(int), 0, grid_h - 1)
            obs_grid[orow, oc] = True

        # CRITICAL: Any cell outside the room interior is an impassable obstacle!
        # This physically prevents the path from exiting the room or going around outer walls!
        occupancy_grid = obs_grid | (~room_interior)

        # 7. Inflate obstacles with robot radius
        inflation_cells = int(np.ceil(self.config.robot_radius / res))
        if inflation_cells > 0:
            y, x = np.ogrid[-inflation_cells : inflation_cells + 1, -inflation_cells : inflation_cells + 1]
            circular_struct = x**2 + y**2 <= inflation_cells**2
            inflated_grid = ndimage.binary_dilation(occupancy_grid, structure=circular_struct)
        else:
            inflated_grid = occupancy_grid.copy()

        return {
            "occupancy_grid": occupancy_grid,
            "inflated_grid": inflated_grid,
            "room_interior": room_interior,
            "origin": (min_x, min_y),
            "resolution": res,
            "rotation_matrix": r_align,
            "ground_z": ground_z,
            "bounds": (min_x, max_x, min_y, max_y),
            "grid_shape": (grid_h, grid_w),
        }
