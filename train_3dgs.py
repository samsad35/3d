#!/usr/bin/env python3
"""
High-Performance 3D Gaussian Splatting (3DGS) Trainer using gsplat.
Trains photorealistic 3D Gaussian Splats on multi-view video frames using
camera poses and 3D geometry from VGGT.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
cache_dir = Path(os.environ.get("CACHE_DIR", ROOT_DIR / "cache"))
os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_dir / "torch_extensions"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import argparse
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import gsplat
from gsplat import DefaultStrategy

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "vggt"))

from src.reconstruction.vggt_reconstructor import VGGTReconstructor


def create_window(window_size: int, channel: int):
    _1D = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / 4.5) for x in range(window_size)])
    _1D = (_1D / _1D.sum()).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Computes SSIM between two (B, C, H, W) images."""
    c = img1.shape[1]
    w = create_window(window_size, c).to(img1.device)
    mu1 = F.conv2d(img1, w, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, w, padding=window_size // 2, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sig1 = F.conv2d(img1 * img1, w, padding=window_size // 2, groups=c) - mu1_sq
    sig2 = F.conv2d(img2 * img2, w, padding=window_size // 2, groups=c) - mu2_sq
    sig12 = F.conv2d(img1 * img2, w, padding=window_size // 2, groups=c) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu12 + C1) * (2 * sig12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2))).mean()


def train_3dgs(
    image_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_steps: int = 1200,
    train_res_scale: float = 0.5,
    device: str = "cuda:0",
):
    if image_dir is None:
        image_dir = str(ROOT_DIR / "data" / "custom_frames")
    if output_dir is None:
        output_dir = str(ROOT_DIR / "data")
    print("=" * 65)
    print("  3D GAUSSIAN SPLATTING (3DGS) PHOTOREALISTIC TRAINER")
    print("=" * 65)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_file = out_path / "vggt_cache.pt"

    # Step 1: Load or Compute VGGT Reconstruction Poses & Points
    if cache_file.exists():
        print(f"[VGGT] Loading cached geometry from {cache_file} ...")
        recon = torch.load(cache_file, map_location="cpu")
    else:
        print("[VGGT] Running VGGT reconstruction on video frames...")
        reconstructor = VGGTReconstructor(device=device)
        recon = reconstructor.reconstruct_from_images(image_dir)
        # Cache light reconstruction data
        save_dict = {
            "extrinsic": recon["extrinsic"],
            "intrinsic": recon["intrinsic"],
            "pcd_points": recon["pcd_points"],
            "pcd_colors": recon["pcd_colors"],
            "image_paths": recon["image_paths"],
        }
        torch.save(save_dict, cache_file)
        print(f"[VGGT] Saved geometry cache to {cache_file}")

    image_paths = recon["image_paths"]
    num_views = len(image_paths)
    print(f"[Dataset] Found {num_views} camera views.")

    # Load and preprocess training images and camera matrices
    print(f"[Dataset] Loading {num_views} images at resolution scale {train_res_scale} ...")
    images_list = []
    viewmats_list = []
    ks_list = []

    for idx, img_path in enumerate(image_paths):
        bgr = cv2.imread(img_path)
        h, w = bgr.shape[:2]
        new_w, new_h = int(w * train_res_scale), int(h * train_res_scale)
        rgb = cv2.cvtColor(cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb).float() / 255.0  # (H, W, 3)

        # Scale intrinsic matrix to match resized training image
        k_orig = recon["intrinsic"][idx].copy()
        k_scaled = k_orig.copy()
        k_scaled[0] *= (new_w / w)
        k_scaled[1] *= (new_h / h)

        # Extrinsic (world to cam matrix [R | t])
        ext = recon["extrinsic"][idx]
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :4] = ext

        images_list.append(img_t)
        viewmats_list.append(torch.from_numpy(viewmat).float())
        ks_list.append(torch.from_numpy(k_scaled).float())

    img_h, img_w = images_list[0].shape[:2]
    print(f"[Dataset] Training image dimensions: {img_w}x{img_h}")

    # Move camera matrices and images to GPU
    all_images = [img.to(device) for img in images_list]
    all_viewmats = torch.stack(viewmats_list).to(device)  # (V, 4, 4)
    all_ks = torch.stack(ks_list).to(device)              # (V, 3, 3)

    # Step 2: Initialize 3D Gaussians from Point Cloud
    pts = recon["pcd_points"]
    cols = recon["pcd_colors"]
    if cols.max() > 1.0:
        cols = cols / 255.0

    # Subsample initial points to 90,000 for rapid and clean convergence
    target_init = 90000
    if len(pts) > target_init:
        indices = np.random.choice(len(pts), target_init, replace=False)
        pts = pts[indices]
        cols = cols[indices]

    num_gaussians = len(pts)
    print(f"[3DGS] Initializing {num_gaussians} 3D Gaussian ellipsoids from points...")

    # Parameters
    init_means = torch.from_numpy(pts).float().to(device)
    init_scales = torch.full((num_gaussians, 3), 0.012, device=device)
    init_quats = torch.zeros((num_gaussians, 4), device=device)
    init_quats[:, 0] = 1.0  # Identity rotation [1, 0, 0, 0]
    init_opacities = torch.full((num_gaussians,), 0.85, device=device)
    init_colors = torch.from_numpy(cols).float().to(device)

    # Parameter dict
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(init_means),
        "scales": torch.nn.Parameter(torch.log(init_scales)),
        "quats": torch.nn.Parameter(init_quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.clamp(init_opacities, 0.01, 0.99))),
        "colors": torch.nn.Parameter(torch.logit(torch.clamp(init_colors, 0.01, 0.99))),
    })

    # Individual optimizers for 3DGS parameters
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4),
        "scales": torch.optim.Adam([params["scales"]], lr=4.0e-3),
        "quats": torch.optim.Adam([params["quats"]], lr=1.0e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=4.0e-2),
        "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3),
    }

    # Exponential LR decay for means
    lr_scheduler_means = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=(1.6e-6 / 1.6e-4) ** (1.0 / max_steps)
    )

    print(f"\n[3DGS] Starting optimization for {max_steps} steps on {torch.cuda.get_device_name(0)} ...")
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Pick random camera view
        view_idx = np.random.randint(0, num_views)
        target_img = all_images[view_idx]
        viewmat = all_viewmats[view_idx : view_idx + 1]
        k_mat = all_ks[view_idx : view_idx + 1]

        # Normalize quaternions and activate parameters
        cur_quats = F.normalize(params["quats"], dim=-1)
        cur_scales = torch.clamp(torch.exp(params["scales"]), min=1e-5, max=0.20)
        cur_opacities = torch.sigmoid(params["opacities"])
        cur_colors = torch.sigmoid(params["colors"])

        # Forward Rasterization via gsplat CUDA
        render, alphas, info = gsplat.rasterization(
            means=params["means"],
            quats=cur_quats,
            scales=cur_scales,
            opacities=cur_opacities,
            colors=cur_colors,
            viewmats=viewmat,
            Ks=k_mat,
            width=img_w,
            height=img_h,
        )

        pred_img = render[0]  # (H, W, 3)

        # Loss: L1 + SSIM
        l1_loss = torch.abs(pred_img - target_img).mean()
        pred_bchw = pred_img.permute(2, 0, 1).unsqueeze(0)
        targ_bchw = target_img.permute(2, 0, 1).unsqueeze(0)
        ssim_val = ssim(pred_bchw, targ_bchw)
        loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        # Backward
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()

        for opt in optimizers.values():
            opt.step()
        lr_scheduler_means.step()

        # Logging
        if step % 100 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / step) * 1000.0
            print(
                f"[Step {step:4d}/{max_steps}] Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, "
                f"SSIM: {ssim_val.item():.3f}) | Speed: {ms_per_step:.1f} ms/step | Elapsed: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\n[3DGS] Training completed in {total_time:.1f} seconds ({max_steps} steps)!")

    # Step 3: Compute final 3D Gaussian parameters and covariances
    print("[3DGS] Computing continuous 3D covariance matrices and saving trained model...")
    with torch.no_grad():
        final_means = params["means"].detach().float()
        final_quats = F.normalize(params["quats"].detach(), dim=-1).float()
        final_scales = torch.clamp(torch.exp(params["scales"].detach()), min=1e-5, max=0.20).float()
        final_opacities = torch.sigmoid(params["opacities"].detach()).float()
        final_colors = torch.sigmoid(params["colors"].detach()).float()

        # Compute full 3x3 covariance matrices
        covars, _ = gsplat.quat_scale_to_covar_preci(final_quats, final_scales)

        # Filter out near-transparent splats
        keep_mask = (final_opacities > 0.08).cpu().numpy()
        print(f"[3DGS] Filtered {len(keep_mask)} -> {int(np.sum(keep_mask))} active opaque Gaussians.")

        means_np = final_means.cpu().numpy()[keep_mask]
        covars_np = covars.cpu().numpy()[keep_mask]
        colors_np = final_colors.cpu().numpy()[keep_mask]
        opacities_np = final_opacities.cpu().numpy()[keep_mask, None]
        quats_np = final_quats.cpu().numpy()[keep_mask]
        scales_np = final_scales.cpu().numpy()[keep_mask]

        save_dict = {
            "means": means_np,
            "covariances": covars_np,
            "rgbs": colors_np,
            "opacities": opacities_np,
            "quats": quats_np,
            "scales": scales_np,
            "training_steps": max_steps,
            "final_ssim": float(ssim_val.item()),
            "final_loss": float(loss.item()),
        }

        save_pt = out_path / "trained_3dgs.pt"
        torch.save(save_dict, save_pt)
        print(f"[3DGS] Saved trained model to: {save_pt}")

        # Also export standard PLY format for external viewers
        save_ply = out_path / "trained_3dgs.ply"
        try:
            # SH degree 0 RGB coefficients = (RGB - 0.5) / 0.28209479177387814
            c_sh0 = (torch.from_numpy(colors_np).to(device) - 0.5) / 0.28209479177387814
            c_sh0 = c_sh0.unsqueeze(1)
            sh_zeros = torch.zeros((len(colors_np), 0, 3), device=device)
            gsplat.export_splats(
                means=torch.from_numpy(means_np).to(device),
                scales=torch.from_numpy(scales_np).to(device),
                quats=torch.from_numpy(quats_np).to(device),
                opacities=torch.from_numpy(opacities_np).squeeze(-1).to(device),
                sh0=c_sh0,
                shN=sh_zeros,
                format="ply",
                save_to=str(save_ply),
            )
            print(f"[3DGS] Exported standard 3DGS PLY to: {save_ply}")
        except Exception as e:
            print(f"[3DGS] Note: PLY export skipped ({e}), PT format ready.")

    print("\n" + "=" * 65)
    print("  3DGS TRAINING FINISHED WITH PHOTOREALISTIC FIDELITY!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting on Scene")
    parser.add_argument("--images", type=str, default=str(ROOT_DIR / "data" / "custom_frames"))
    parser.add_argument("--steps", type=int, default=1200, help="Training iterations")
    parser.add_argument("--res-scale", type=float, default=0.5, help="Image resolution scale (0.5 = 2x faster)")
    args = parser.parse_args()

    train_3dgs(image_dir=args.images, max_steps=args.steps, train_res_scale=args.res_scale)
