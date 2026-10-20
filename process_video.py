#!/usr/bin/env python3
"""
Full End-to-End Pipeline CLI:
1. Video Keyframe Extraction (OpenCV)
2. VGGT 3D Metric Dense Reconstruction (Meta AI)
3. 2D Ground Plane Alignment & Occupancy Grid Mapping
4. Grounded-SAM 3D Object Detection & Lifting
5. Metric Colorized Depth Maps Export (Turbo Colormap)
6. Complete Scene Cache Export for Instant Visualizer Startup
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent
default_cache = Path(os.environ.get("CACHE_DIR", ROOT_DIR / "cache"))
os.environ.setdefault("HF_HOME", str(default_cache / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(default_cache / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(default_cache / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(default_cache))

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "vggt"))

import cv2
import numpy as np
import torch
from src.pipeline import SceneNavigator
from extract_video_frames import extract_keyframes


DEFAULT_CLASSES = [
    "bed", "chair", "table", "desk", "sofa", "couch", "pillow",
    "lamp", "laptop", "monitor", "backpack", "bottle", "door", "window",
    "plant", "shelf", "box", "keyboard", "mouse", "cabinet", "tv", "closet"
]


def process_scene_pipeline(
    video_path: Optional[str] = None,
    images_dir: Optional[str] = None,
    num_frames: int = 80,
    max_dim: int = 1024,
    cache_output: str = "data/scene_cache.pt",
    depth_output: str = "data/depth_maps",
    run_segmentation: bool = True,
    classes: Optional[List[str]] = None,
    sam_subsample: int = 1,
):
    print("=" * 70)
    print("  3D SCENE UNDERSTANDING & ARCHITECTURAL RECONSTRUCTION PIPELINE")
    print("=" * 70)

    # 1. Video extraction if video provided
    resolved_images_dir = images_dir
    if video_path and Path(video_path).exists():
        if not resolved_images_dir:
            resolved_images_dir = str(ROOT_DIR / "data" / "extracted_frames")
        print(f"\n[Step 1/4] Extracting {num_frames} keyframes from: {video_path}")
        extract_keyframes(
            video_path=video_path,
            output_dir=resolved_images_dir,
            num_frames=num_frames,
            max_dim=max_dim,
        )
    elif not resolved_images_dir:
        resolved_images_dir = str(ROOT_DIR / "data" / "custom_frames")

    img_path = Path(resolved_images_dir)
    frames = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))
    print(f"\n[Input] Found {len(frames)} images in {img_path}")
    if len(frames) == 0:
        raise FileNotFoundError(f"No image files found in {img_path}")

    # 2. Run Pipeline
    target_classes = classes or DEFAULT_CLASSES
    navigator = SceneNavigator()

    print("\n[Step 2/4] Running VGGT 3D Reconstruction and Occupancy Grid Mapping...")
    navigator.process_scene(
        image_inputs=str(img_path),
        object_classes=target_classes,
        run_segmentation=run_segmentation,
    )

    # 3. Save Scene Cache
    resolved_cache = Path(cache_output)
    if not resolved_cache.is_absolute():
        resolved_cache = ROOT_DIR / resolved_cache
    resolved_cache.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 3/4] Saving scene cache to: {resolved_cache} ...")
    torch.save({
        "reconstruction_data": navigator.reconstruction_data,
        "grid_data": navigator.grid_data,
        "detected_objects": navigator.detected_objects,
        "fixed_start_3d": navigator.fixed_start_3d,
        "fixed_start_cell": navigator.fixed_start_cell,
    }, str(resolved_cache))
    print(f"  ✓ Cache saved successfully ({resolved_cache.stat().st_size / 1e6:.1f} MB)!")

    # 4. Export Depth Maps
    resolved_depth = Path(depth_output)
    if not resolved_depth.is_absolute():
        resolved_depth = ROOT_DIR / resolved_depth
    resolved_depth.mkdir(parents=True, exist_ok=True)

    depth_tensor = navigator.reconstruction_data["depth"]
    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.detach().cpu().numpy()
    else:
        depth_np = np.array(depth_tensor)

    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np.squeeze(-1)

    print(f"\n[Step 4/4] Exporting {len(depth_np)} colorized depth maps to: {resolved_depth} ...")
    for i in range(len(depth_np)):
        d = depth_np[i]
        np.save(str(resolved_depth / f"depth_raw_{i:03d}.npy"), d)

        valid = d > 0.05
        if np.sum(valid) > 10:
            d_min = np.percentile(d[valid], 2)
            d_max = np.percentile(d[valid], 98)
            d_norm = np.clip((d - d_min) / max(1e-4, (d_max - d_min)), 0.0, 1.0)
        else:
            d_norm = np.zeros_like(d)

        d_u8 = (d_norm * 255.0).astype(np.uint8)
        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(resolved_depth / f"depth_{i:03d}.png"), d_color)

    print(f"  ✓ {len(depth_np)} depth maps exported successfully!")
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE! You can now launch:")
    print("  • 3D Navigator:   python app.py")
    print("  • App-Architect:  python app_architect.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="End-to-end 3D Scene Reconstruction & Mapping")
    parser.add_argument("--video", type=str, default=None, help="Path to input raw video file (e.g. .mp4)")
    parser.add_argument("--images", type=str, default=None, help="Path to existing frames directory")
    parser.add_argument("--num-frames", type=int, default=80, help="Number of keyframes to extract from video")
    parser.add_argument("--max-dim", type=int, default=1024, help="Max image dimension for downscaling")
    parser.add_argument("--output-cache", type=str, default="data/scene_cache.pt", help="Output path for .pt cache")
    parser.add_argument("--output-depth", type=str, default="data/depth_maps", help="Output folder for depth maps")
    parser.add_argument("--no-segment", action="store_true", help="Skip Grounded-SAM object segmentation")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated target object classes")

    args = parser.parse_args()
    classes_list = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None

    process_scene_pipeline(
        video_path=args.video,
        images_dir=args.images,
        num_frames=args.num_frames,
        max_dim=args.max_dim,
        cache_output=args.output_cache,
        depth_output=args.output_depth,
        run_segmentation=not args.no_segment,
        classes=classes_list,
    )


if __name__ == "__main__":
    main()
