"""
End-to-End Scene Understanding and Object Navigation Pipeline
Combines VGGT 3D reconstruction, Grounded-SAM object extraction,
2D occupancy grid mapping, and A* path planning to any requested object.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.reconstruction.vggt_reconstructor import VGGTReconstructor
from src.mapping.occupancy_grid import OccupancyGridGenerator, GridMapConfig
from src.segmentation.grounded_sam import GroundedSAMSegmenter, DetectedObject3D
from src.planning.path_planner import AStarPlanner


@dataclass
class NavigationResult:
    target_object: DetectedObject3D
    start_world_3d: np.ndarray
    goal_world_3d: np.ndarray
    path_grid: List[Tuple[int, int]]
    path_world_3d: np.ndarray       # (K, 3) 3D waypoints
    smoothed_path_3d: np.ndarray   # (M, 3) smoothed 3D waypoints
    occupancy_map_image: np.ndarray


class SceneNavigator:
    def __init__(
        self,
        reconstructor: Optional[VGGTReconstructor] = None,
        segmenter: Optional[GroundedSAMSegmenter] = None,
        map_config: Optional[GridMapConfig] = None,
    ):
        self.reconstructor = reconstructor
        self.segmenter = segmenter
        self.grid_generator = OccupancyGridGenerator(map_config or GridMapConfig())
        self.planner = AStarPlanner(allow_diagonal=True)

        self.reconstruction_data: Optional[Dict] = None
        self.grid_data: Optional[Dict] = None
        self.detected_objects: List[DetectedObject3D] = []
        self.fixed_start_cell: Tuple[int, int] = (0, 0)
        self.fixed_start_3d: np.ndarray = np.array([0.0, 0.0, 0.0])

    def load_cache(self, cache_path: Union[str, Path]):
        """Directly loads precomputed reconstruction, grid, and objects from disk."""
        print(f"[SceneNavigator] Loading scene from cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.reconstruction_data = cache["reconstruction_data"]
        self.grid_data = cache["grid_data"]
        self.detected_objects = cache["detected_objects"]

        if "camera_centers" not in self.reconstruction_data:
            extrinsics = self.reconstruction_data["extrinsic"]
            cam_centers = []
            for ext in extrinsics:
                r = ext[:, :3]
                t = ext[:, 3]
                cam_centers.append(-r.T @ t)
            self.reconstruction_data["camera_centers"] = np.array(cam_centers)

        occ_grid = self.grid_data["occupancy_grid"]
        main_walkable = AStarPlanner.get_main_walkable_mask(occ_grid)
        walkable_cells = np.argwhere(main_walkable)
        if len(walkable_cells) > 0:
            min_r = int(np.min(walkable_cells[:, 0]))
            matching_cols = walkable_cells[walkable_cells[:, 0] == min_r, 1]
            median_c = int(np.median(matching_cols))
            self.fixed_start_cell = (min_r, median_c)

            res = self.grid_data["resolution"]
            min_x, min_y = self.grid_data["origin"]
            ground_z = self.grid_data["ground_z"]
            r_inv = np.linalg.inv(self.grid_data["rotation_matrix"])
            start_al = np.array([min_x + median_c * res, min_y + min_r * res, ground_z + 0.22])
            self.fixed_start_3d = r_inv @ start_al
        else:
            self.fixed_start_cell = (0, 0)
            self.fixed_start_3d = np.array([0.0, 0.0, 0.0])
        print(f"[SceneNavigator] Cache loaded: {len(self.detected_objects)} objects, grid shape: {self.grid_data['grid_shape']}")

    def process_scene(
        self,
        image_inputs: Union[str, List[str]],
        object_classes: Optional[List[str]] = None,
        run_segmentation: bool = True,
    ):
        """
        Runs complete 3D scene reconstruction, occupancy grid mapping, and object discovery.
        """
        if self.reconstructor is None:
            self.reconstructor = VGGTReconstructor()
        if object_classes is None:
            object_classes = [
                "bed", "chair", "table", "desk", "sofa", "pillow",
                "lamp", "laptop", "monitor", "backpack", "bottle", "door", "window"
            ]

        # 1. VGGT 3D Reconstruction
        print("\n==========================================")
        print("[Step 1/3] Running VGGT 3D Reconstruction...")
        print("==========================================")
        self.reconstruction_data = self.reconstructor.reconstruct_from_images(image_inputs)

        # Compute camera centers in world coordinates: C = - R^T @ t
        extrinsics = self.reconstruction_data["extrinsic"]
        cam_centers = []
        for ext in extrinsics:
            r = ext[:, :3]
            t = ext[:, 3]
            cam_centers.append(-r.T @ t)
        cam_centers = np.array(cam_centers)
        self.reconstruction_data["camera_centers"] = cam_centers

        # 2. 2D Occupancy Grid Generation
        print("\n==========================================")
        print("[Step 2/3] Generating 2D Occupancy Grid...")
        print("==========================================")
        self.grid_data = self.grid_generator.generate_grid_map(
            points=self.reconstruction_data["pcd_points"],
            camera_centers=cam_centers,
        )
        print(f"[Map] Grid Resolution: {self.grid_data['resolution']} m/cell")
        print(f"[Map] Grid Shape: {self.grid_data['grid_shape'][1]}x{self.grid_data['grid_shape'][0]} cells")

        # Compute a SINGLE PERMANENT STARTING POINT at the room entrance
        occ_grid = self.grid_data["occupancy_grid"]
        main_walkable = AStarPlanner.get_main_walkable_mask(occ_grid)
        walkable_cells = np.argwhere(main_walkable)
        if len(walkable_cells) > 0:
            min_r = int(np.min(walkable_cells[:, 0]))
            matching_cols = walkable_cells[walkable_cells[:, 0] == min_r, 1]
            median_c = int(np.median(matching_cols))
            self.fixed_start_cell = (min_r, median_c)

            res = self.grid_data["resolution"]
            min_x, min_y = self.grid_data["origin"]
            ground_z = self.grid_data["ground_z"]
            r_inv = np.linalg.inv(self.grid_data["rotation_matrix"])
            start_al = np.array([min_x + median_c * res, min_y + min_r * res, ground_z + 0.22])
            self.fixed_start_3d = r_inv @ start_al
        else:
            self.fixed_start_cell = (0, 0)
            self.fixed_start_3d = np.array([0.0, 0.0, 0.0])

        print(f"[Map] Fixed room entrance set at: {np.round(self.fixed_start_3d, 2)} (cell: {self.fixed_start_cell})")

        # 3. Grounded-SAM Object Detection & 3D Lifting
        if run_segmentation:
            print("\n==========================================")
            print("[Step 3/3] Segmenting & Extracting 3D Objects...")
            print("==========================================")
            if self.segmenter is None:
                self.segmenter = GroundedSAMSegmenter()

            world_pts = self.reconstruction_data["computed_world_points"]
            conf = self.reconstruction_data["depth_conf"]
            paths = self.reconstruction_data["image_paths"]

            self.detected_objects = self.segmenter.lift_detections_to_3d(
                image_paths=paths,
                world_points_map=world_pts,
                confidence_map=conf,
                text_queries=object_classes,
            )
            print(f"[Scene] Found {len(self.detected_objects)} objects:")
            for idx, obj in enumerate(self.detected_objects):
                c = obj.centroid_3d
                print(f"  [{idx}] '{obj.label}' (conf: {obj.confidence:.2f}) at 3D: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

        print("\n[SceneNavigator] Scene processing complete!\n")

    def find_object(self, query: str) -> Optional[DetectedObject3D]:
        """
        Finds the best matching object for a query string.
        """
        if not self.detected_objects:
            print("[SceneNavigator] No objects detected in scene.")
            return None

        q = query.lower().strip()
        candidates = []
        for obj in self.detected_objects:
            label = obj.label.lower()
            if q in label or label in q:
                candidates.append(obj)

        if not candidates:
            # Fallback: substring matching
            for obj in self.detected_objects:
                for word in q.split():
                    if word in obj.label.lower():
                        candidates.append(obj)
                        break

        if candidates:
            # Pick highest confidence match
            return max(candidates, key=lambda x: x.confidence)

        return None

    def navigate_to_object(
        self,
        object_query: str,
        start_world_pos: Optional[np.ndarray] = None,
    ) -> Optional[NavigationResult]:
        """
        Plans a path to the requested object from the start position (default: Camera 0).
        """
        if self.grid_data is None:
            raise RuntimeError("Please call process_scene() before navigating.")

        target = self.find_object(object_query)
        if target is None:
            print(f"[Navigation] Object '{object_query}' not found in the scene.")
            print(f"Available objects: {[o.label for o in self.detected_objects]}")
            return None

        print(f"[Navigation] Target object found: '{target.label}' (conf {target.confidence:.2f})")

        # 1. Transform start position to aligned grid coordinates
        r_align = self.grid_data["rotation_matrix"]
        res = self.grid_data["resolution"]
        min_x, min_y = self.grid_data["origin"]
        ground_z = self.grid_data["ground_z"]

        occ_grid = self.grid_data["occupancy_grid"]

        # 2. Transform target centroid to aligned grid coordinates
        target_aligned = r_align @ target.centroid_3d
        target_c = int((target_aligned[0] - min_x) / res)
        target_r = int((target_aligned[1] - min_y) / res)

        # 1. Determine start cell inside main room floor
        main_walkable = AStarPlanner.get_main_walkable_mask(occ_grid)

        if start_world_pos is not None:
            start_aligned = r_align @ start_world_pos
            sc = int((start_aligned[0] - min_x) / res)
            sr = int((start_aligned[1] - min_y) / res)
            start_cell = AStarPlanner.find_nearest_navigable_cell(occ_grid, (sr, sc), valid_mask=main_walkable)
            start_r, start_c = start_cell
        else:
            # Use the permanent fixed entrance point (constant for all queries!)
            start_cell = self.fixed_start_cell
            start_r, start_c = start_cell
            start_world_pos = self.fixed_start_3d

        # 3. Plan A* path strictly within main floor
        raw_path = self.planner.plan_path(
            grid=occ_grid,
            start_cell=(start_r, start_c),
            goal_cell=(target_r, target_c),
            safe_clearance_cells=2.0,
        )

        if not raw_path:
            print(f"[Navigation] Could not find a safe path to '{object_query}'.")
            return None

        smoothed_path = self.planner.smooth_path(occ_grid, raw_path)
        print(f"[Navigation] Path found! {len(raw_path)} steps (smoothed: {len(smoothed_path)} waypoints).")

        # 4. Convert 2D grid path back to 3D world coordinates
        r_inv = np.linalg.inv(r_align)

        def grid_to_world_3d(path_cells):
            pts_3d = []
            for r, c in path_cells:
                # Aligned coordinates elevated +0.22m above ground (clearly visible above floor)
                x_al = min_x + c * res
                y_al = min_y + r * res
                z_al = ground_z + 0.22
                pt_al = np.array([x_al, y_al, z_al])
                pt_world = r_inv @ pt_al
                pts_3d.append(pt_world)
            return np.array(pts_3d)

        path_world_3d = grid_to_world_3d(raw_path)
        smoothed_world_3d = grid_to_world_3d(smoothed_path)

        # 5. Render Visualization Map
        vis_map = self._render_map(raw_path, smoothed_path, (start_r, start_c), (target_r, target_c), target)

        return NavigationResult(
            target_object=target,
            start_world_3d=start_world_pos,
            goal_world_3d=smoothed_world_3d[-1] if len(smoothed_world_3d) > 0 else target.centroid_3d,
            path_grid=raw_path,
            path_world_3d=path_world_3d,
            smoothed_path_3d=smoothed_world_3d,
            occupancy_map_image=vis_map,
        )

    def _render_map(
        self,
        raw_path: List[Tuple[int, int]],
        smoothed_path: List[Tuple[int, int]],
        start_cell: Tuple[int, int],
        target_cell: Tuple[int, int],
        target_obj: Optional[DetectedObject3D] = None,
        target_width: int = 500,
    ) -> np.ndarray:
        h, w = self.grid_data["grid_shape"]
        occ_grid = self.grid_data["occupancy_grid"]
        inflated_grid = self.grid_data["inflated_grid"]

        # Base grid image (Y down in numpy, we flip later)
        vis = np.ones((h, w, 3), dtype=np.uint8) * 255
        vis[inflated_grid] = [215, 215, 215]  # Light gray inflation
        vis[occ_grid] = [45, 45, 45]          # Dark obstacles

        # Flip vertically to match standard cartesian coords (Y up)
        vis = np.flipud(vis)

        # Scale up to crisp high resolution
        scale = max(2, target_width // w)
        out_w = w * scale
        out_h = h * scale
        large_map = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # Coordinate helper: grid (row, col) -> large_map (px_x, px_y)
        def to_px(r, c):
            px_y = int((h - 1 - r + 0.5) * scale)
            px_x = int((c + 0.5) * scale)
            return px_x, px_y

        # Draw detected objects (purple markers)
        r_align = self.grid_data["rotation_matrix"]
        res = self.grid_data["resolution"]
        min_x, min_y = self.grid_data["origin"]

        for obj in self.detected_objects:
            obj_al = r_align @ obj.centroid_3d
            oc = int((obj_al[0] - min_x) / res)
            orow = int((obj_al[1] - min_y) / res)
            if 0 <= orow < h and 0 <= oc < w:
                px_x, px_y = to_px(orow, oc)
                cv2.circle(large_map, (px_x, px_y), max(4, scale - 1), (180, 0, 180), -1)
                cv2.putText(
                    large_map, obj.label, (px_x + 5, px_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 0, 110), 1, cv2.LINE_AA
                )

        # Draw raw path (red line)
        if raw_path:
            pts = [to_px(r, c) for r, c in raw_path]
            for i in range(len(pts) - 1):
                cv2.line(large_map, pts[i], pts[i + 1], (0, 0, 255), max(2, scale // 2))

        # Draw smoothed waypoints (blue circles)
        if smoothed_path:
            for r, c in smoothed_path:
                px_x, px_y = to_px(r, c)
                cv2.circle(large_map, (px_x, px_y), max(3, scale // 2), (255, 0, 0), -1)

        # Start position (Green circle)
        if start_cell != (0, 0):
            sx, sy = to_px(start_cell[0], start_cell[1])
            cv2.circle(large_map, (sx, sy), max(6, scale), (0, 200, 0), -1)
            cv2.circle(large_map, (sx, sy), max(7, scale + 1), (0, 100, 0), 1)
            cv2.putText(large_map, "Depart", (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 130, 0), 1, cv2.LINE_AA)

        # Target position (Orange circle)
        if target_cell != (0, 0):
            tx, ty = to_px(target_cell[0], target_cell[1])
            cv2.circle(large_map, (tx, ty), max(7, scale + 1), (0, 140, 255), -1)
            cv2.circle(large_map, (tx, ty), max(8, scale + 2), (0, 90, 200), 1)
            label = target_obj.label if target_obj else "Cible"
            cv2.putText(large_map, label, (tx + 8, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)

        # Add top banner with title & legend
        banner_h = 32
        banner = np.ones((banner_h, out_w, 3), dtype=np.uint8) * 242
        cv2.putText(banner, "Plan 2D: [Vert: Depart | Rouge: Trajet | Orange: Cible | Violet: Objets]", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)

        return np.vstack([banner, large_map])
