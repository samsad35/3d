"""
A* 2D Path Planner Module with Connected Component Guarantee & Costmap
Computes safe, collision-free navigable trajectories strictly within the room floor.
"""

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, label


class AStarPlanner:
    def __init__(self, allow_diagonal: bool = True):
        self.allow_diagonal = allow_diagonal
        if allow_diagonal:
            # 8-connected motion: (dx, dy, cost)
            self.motions = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (-1, -1, math.sqrt(2)),
            ]
        else:
            self.motions = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
            ]

    @staticmethod
    def euclidean_heuristic(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def get_main_walkable_mask(occupancy_grid: np.ndarray) -> np.ndarray:
        """
        Extracts the largest connected component of traversable free floor space.
        """
        labeled, num = label(~occupancy_grid)
        if num == 0:
            return ~occupancy_grid
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        largest_id = int(np.argmax(sizes) + 1)
        return labeled == largest_id

    @classmethod
    def find_nearest_navigable_cell(
        cls,
        occupancy_grid: np.ndarray,
        cell: Tuple[int, int],
        valid_mask: Optional[np.ndarray] = None,
        max_radius: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """
        Finds the closest cell strictly inside the valid main room floor.
        """
        h, w = occupancy_grid.shape
        r = int(np.clip(cell[0], 0, h - 1))
        c = int(np.clip(cell[1], 0, w - 1))

        if valid_mask is None:
            valid_mask = cls.get_main_walkable_mask(occupancy_grid)

        if valid_mask[r, c]:
            return (r, c)

        # Search outward in concentric rings
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc]:
                        return (nr, nc)
            for dc in range(-radius + 1, radius):
                for dr in (-radius, radius):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc]:
                        return (nr, nc)

        # Fallback to closest valid cell
        free_cells = np.argwhere(valid_mask)
        if len(free_cells) > 0:
            dists = np.linalg.norm(free_cells - np.array([r, c]), axis=1)
            idx = np.argmin(dists)
            return (int(free_cells[idx][0]), int(free_cells[idx][1]))

        return None

    def plan_path(
        self,
        grid: np.ndarray,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        safe_clearance_cells: float = 2.0,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Plans collision-free path strictly within the main walkable room floor.
        """
        h, w = grid.shape
        main_walkable = self.get_main_walkable_mask(grid)

        start = self.find_nearest_navigable_cell(grid, start_cell, valid_mask=main_walkable)
        goal = self.find_nearest_navigable_cell(grid, goal_cell, valid_mask=main_walkable)

        if start is None or goal is None:
            print("[A* Planner] Unable to find valid start or goal cell inside main room floor.")
            return None

        if start == goal:
            return [start]

        dist_transform = distance_transform_edt(~grid)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.euclidean_heuristic(start, goal)}
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            cr, cc = current
            for dr, dc, base_cost in self.motions:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                # Strictly impassable if outside grid, outside main floor, or on an obstacle
                if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] or not main_walkable[nr, nc]:
                    continue

                clearance = dist_transform[nr, nc]
                penalty = max(0.0, (safe_clearance_cells - clearance)) * 2.0

                tentative_g = g_score[current] + base_cost + penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.euclidean_heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        print("[A* Planner] No path could be found between start and goal.")
        return None

    @staticmethod
    def smooth_path(grid: np.ndarray, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Shortcuts redundant waypoints using line-of-sight raycasting.
        """
        if len(path) <= 2:
            return path

        def line_of_sight(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
            r0, c0 = p1
            r1, c1 = p2
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            sr = 1 if r0 < r1 else -1
            sc = 1 if c0 < c1 else -1
            err = dr - dc

            cr, cc = r0, c0
            while True:
                if grid[cr, cc]:
                    return False
                if cr == r1 and cc == c1:
                    break
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc
                    cr += sr
                if e2 < dr:
                    err += dr
                    cc += sc
            return True

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > current_idx + 1:
                if line_of_sight(path[current_idx], path[next_idx]):
                    break
                next_idx -= 1
            smoothed.append(path[next_idx])
            current_idx = next_idx

        return smoothed
