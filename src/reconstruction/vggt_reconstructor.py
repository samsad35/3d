"""
VGGT 3D Reconstructor Module
Extracts 3D point cloud, depth maps, and camera poses from multi-view images using VGGT.
"""

import os
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
default_cache = ROOT_DIR / "cache"
os.environ.setdefault("HF_HOME", str(default_cache / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(default_cache / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(default_cache / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(default_cache))

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import sys
# Add vggt to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTReconstructor:
    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if dtype is None:
            if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

        print(f"[VGGT] Initializing model '{model_name}' on {self.device} ({self.dtype})...")
        self.model = VGGT.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[VGGT] Model loaded successfully.")

    @torch.no_grad()
    def reconstruct_from_images(
        self,
        image_inputs: Union[str, List[str]],
        conf_percentile_cutoff: float = 30.0,
        use_point_map: bool = False,
    ) -> Dict:
        """
        Runs 3D reconstruction on a set of multi-view images.

        Args:
            image_inputs: Folder path containing images or list of image file paths.
            conf_percentile_cutoff: Filter out lowest X% confidence points (0-100).
            use_point_map: If True, uses direct world_points; if False, unprojects depth.

        Returns:
            dict containing:
                - 'images': (S, 3, H, W) normalized numpy array
                - 'world_points': (S, H, W, 3) 3D coordinate map per view
                - 'depth': (S, H, W, 1) depth map per view
                - 'depth_conf': (S, H, W) confidence map
                - 'extrinsic': (S, 3, 4) camera extrinsics
                - 'intrinsic': (S, 3, 3) camera intrinsics
                - 'pcd_points': (N, 3) filtered aggregated 3D points
                - 'pcd_colors': (N, 3) RGB colors [0, 1] for points
                - 'image_paths': list of source image paths
        """
        if isinstance(image_inputs, str):
            image_dir = Path(image_inputs)
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            image_paths = []
            for ext in extensions:
                image_paths.extend(glob.glob(str(image_dir / ext)))
            image_paths = sorted(image_paths)
        else:
            image_paths = list(image_inputs)

        if not image_paths:
            raise ValueError(f"No valid images found in: {image_inputs}")

        print(f"[VGGT] Processing {len(image_paths)} images...")
        images_tensor = load_and_preprocess_images(image_paths).to(self.device)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.model(images_tensor)

        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # If batch dimension exists, squeeze it
        images_np = images_tensor.detach().cpu().numpy()
        if images_np.ndim == 5:
            images_np = images_np.squeeze(0)

        def get_np(tensor):
            arr = tensor.detach().cpu().float().numpy()
            return arr.squeeze(0) if arr.shape[0] == 1 else arr

        # Move predictions to CPU numpy arrays
        pred_dict = {
            "images": images_np,
            "world_points": get_np(predictions["world_points"]),
            "world_points_conf": get_np(predictions["world_points_conf"]),
            "depth": get_np(predictions["depth"]),
            "depth_conf": get_np(predictions["depth_conf"]),
            "extrinsic": get_np(predictions["extrinsic"]),
            "intrinsic": get_np(predictions["intrinsic"]),
            "image_paths": image_paths,
        }

        # Compute world points from depth if requested (often cleaner)
        if not use_point_map:
            world_pts = unproject_depth_map_to_point_map(
                pred_dict["depth"], pred_dict["extrinsic"], pred_dict["intrinsic"]
            )
            conf = pred_dict["depth_conf"]
        else:
            world_pts = pred_dict["world_points"]
            conf = pred_dict["world_points_conf"]

        pred_dict["computed_world_points"] = world_pts

        # Filter points by confidence threshold
        conf_flat = conf.reshape(-1)
        thresh_val = np.percentile(conf_flat, conf_percentile_cutoff)
        valid_mask = conf_flat >= thresh_val

        # Extract colored point cloud
        # images is (S, 3, H, W) -> transpose to (S, H, W, 3)
        colors = pred_dict["images"].transpose(0, 2, 3, 1).reshape(-1, 3)
        points_flat = world_pts.reshape(-1, 3)

        pred_dict["pcd_points"] = points_flat[valid_mask]
        pred_dict["pcd_colors"] = colors[valid_mask]
        pred_dict["valid_mask"] = valid_mask

        print(
            f"[VGGT] Reconstruction done! Retained {len(pred_dict['pcd_points'])} 3D points "
            f"(filtered lowest {conf_percentile_cutoff}% confidence)."
        )
        return pred_dict

    @staticmethod
    def save_ply(points: np.ndarray, colors: np.ndarray, output_path: str):
        """
        Saves point cloud to standard PLY format.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        num_points = len(points)
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(output_path, "w") as f:
            f.write(header)
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        print(f"[VGGT] Point cloud saved to: {output_path}")
