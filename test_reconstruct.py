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

from src.reconstruction.vggt_reconstructor import VGGTReconstructor

def main():
    room_dir = str(ROOT_DIR / "vggt" / "examples" / "room" / "images")
    output_ply = str(ROOT_DIR / "data" / "room_pcd.ply")

    print("=== Testing VGGT 3D Reconstruction ===")
    reconstructor = VGGTReconstructor()
    
    # Run reconstruction on room images
    result = reconstructor.reconstruct_from_images(
        image_inputs=room_dir,
        conf_percentile_cutoff=30.0,
        use_point_map=False,
    )

    print(f"Computed world points shape: {result['computed_world_points'].shape}")
    print(f"Extrinsics shape: {result['extrinsic'].shape}")
    print(f"Intrinsics shape: {result['intrinsic'].shape}")
    print(f"Filtered point cloud size: {len(result['pcd_points'])} points")

    # Save to PLY
    reconstructor.save_ply(result["pcd_points"], result["pcd_colors"], output_ply)
    print("=== Reconstruction Test Finished Successfully ===")

if __name__ == "__main__":
    main()
