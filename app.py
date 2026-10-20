#!/usr/bin/env python3
"""
Interactive 3D Multi-View Scene Navigator Application
Reconstructs 3D scenes with VGGT, extracts objects with Grounded-SAM,
renders with 3D Gaussian Splatting (3DGS), projects to 2D occupancy grids,
and plans paths to queried objects via Viser WebUI.
"""

import argparse
import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Workspace & Cache path setup (portable for any environment)
ROOT_DIR = Path(__file__).resolve().parent
cache_dir = Path(os.environ.get("CACHE_DIR", ROOT_DIR / "cache"))
os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "vggt"))

import cv2
import numpy as np
import torch
import viser
import viser.transforms as viser_tf

from src.pipeline import SceneNavigator, NavigationResult
from src.mapping.occupancy_grid import GridMapConfig


def make_minimap_html(map_img: np.ndarray) -> str:
    """Creates a sleek floating HUD minimap pinned to the top-left corner."""
    _, buf = cv2.imencode(".png", map_img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return (
        '<div style="position: fixed; top: 18px; left: 18px; z-index: 9999; '
        'background: rgba(255, 255, 255, 0.94); padding: 10px; border-radius: 12px; '
        'box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28); border: 1px solid rgba(0, 0, 0, 0.12); '
        'font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; pointer-events: auto;">'
        '<div style="font-weight: 700; font-size: 13px; color: #111; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">'
        '<span>🗺️ PLAN 2D FIXE (TOP-LEFT)</span>'
        '</div>'
        f'<img src="data:image/png;base64,{b64}" style="width: 280px; height: auto; display: block; border-radius: 6px; border: 1px solid #ddd;"/>'
        '<div style="font-size: 10.5px; color: #444; margin-top: 6px; text-align: center; font-weight: 500;">'
        '🟩 Départ fixe &nbsp;|&nbsp; 🟥 Trajet &nbsp;|&nbsp; 🟧 Cible'
        '</div>'
        '</div>'
    )


def run_app(
    image_dir: str,
    port: int = 8080,
    run_segmentation: bool = True,
    classes_str: str = "bed, chair, table, desk, sofa, pillow, lamp, laptop, monitor, backpack, bottle, door, window",
    min_confidence: float = 0.50,
    cache_path: Optional[str] = "data/scene_cache.pt",
):
    print("=" * 60)
    print("  3D SCENE NAVIGATOR - VGGT + 3D GAUSSIAN SPLATTING + A*")
    print("=" * 60)

    classes = [c.strip() for c in classes_str.split(",") if c.strip()]
    navigator = SceneNavigator()

    # Step 1: Process scene or load from cache
    resolved_cache = Path(cache_path) if cache_path else None
    if resolved_cache and not resolved_cache.is_absolute():
        resolved_cache = ROOT_DIR / resolved_cache

    if resolved_cache and resolved_cache.exists():
        print(f"\n[App] Loading scene from cache: {resolved_cache}")
        navigator.load_cache(resolved_cache)
    else:
        print(f"\n[App] Processing scene images from: {image_dir}")
        navigator.process_scene(
            image_inputs=image_dir,
            object_classes=classes,
            run_segmentation=run_segmentation,
        )

    # Filter detected objects by confidence to avoid cluttering the 3D scene
    raw_objects = navigator.detected_objects
    high_conf_objects = [obj for obj in raw_objects if obj.confidence >= min_confidence]
    if len(high_conf_objects) < 3:
        high_conf_objects = sorted(raw_objects, key=lambda x: x.confidence, reverse=True)[:8]

    # Deduplicate by label keeping highest confidence
    label_map = {}
    for obj in sorted(high_conf_objects, key=lambda x: x.confidence, reverse=True):
        clean_lbl = obj.label.strip().lower()
        if clean_lbl not in label_map:
            label_map[clean_lbl] = obj
    clean_objects = list(label_map.values())

    print(f"[Viser] Filtered {len(raw_objects)} raw detections -> {len(clean_objects)} clean high-confidence objects:")
    for obj in clean_objects:
        print(f"  • {obj.label} (conf: {obj.confidence:.2f}) at 3D: {np.round(obj.centroid_3d, 2)}")

    navigator.detected_objects = clean_objects

    # Step 2: Start Viser Web Server
    print(f"\n[Viser] Starting 3D Web Visualizer on http://0.0.0.0:{port} ...")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None)

    # Step 3: Prepare High-Resolution 3D Point Cloud from Dense Reconstruction
    pcd_pts = navigator.reconstruction_data["pcd_points"]
    pcd_cols = navigator.reconstruction_data["pcd_colors"]
    if pcd_cols.max() <= 1.0:
        pcd_cols_uint8 = (pcd_cols * 255.0).astype(np.uint8)
    else:
        pcd_cols_uint8 = pcd_cols.astype(np.uint8)

    total_pts = len(pcd_pts)
    default_num_points = min(300000, total_pts)
    step = max(1, total_pts // default_num_points)
    initial_pts = pcd_pts[::step]
    initial_cols = pcd_cols_uint8[::step]
    default_point_size = 0.012

    print(f"[Viser] Rendering {len(initial_pts)} sharp, ultra-clean 3D points (from {total_pts} total)...")
    pcd_handle = server.scene.add_point_cloud(
        "/scene/point_cloud",
        points=initial_pts,
        colors=initial_cols,
        point_size=default_point_size,
        point_shape="circle",
        visible=True,
    )

    # Step 3.5: Optional 3DGS layer (kept hidden by default to avoid blurriness)
    trained_3dgs_file = ROOT_DIR / "data" / "trained_3dgs.pt"
    splat_handle = None
    if trained_3dgs_file.exists():
        print(f"[Viser] Loading 3DGS from {trained_3dgs_file} ...")
        gs_data = torch.load(trained_3dgs_file, map_location="cpu", weights_only=False)
        means_3dgs = gs_data["means"].astype(np.float32)
        covs_3dgs = gs_data["covariances"].astype(np.float32)
        rgbs_3dgs = gs_data["rgbs"].astype(np.float32)
        opacities_3dgs = gs_data["opacities"].astype(np.float32)

        splat_handle = server.scene.add_gaussian_splats(
            "/scene/gaussian_splats",
            centers=means_3dgs,
            covariances=covs_3dgs,
            rgbs=rgbs_3dgs,
            opacities=opacities_3dgs,
            visible=False,
        )

    # Step 4: Add Camera Frustums (Hidden by default for a clean view)
    extrinsics = navigator.reconstruction_data["extrinsic"]
    intrinsics = navigator.reconstruction_data["intrinsic"]
    img_h, img_w = navigator.reconstruction_data["images"].shape[-2:]

    cam_handles = []
    for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        R = ext[:, :3]
        t = ext[:, 3]
        cam_pos = -R.T @ t
        R_c2w = R.T
        q_wxyz = viser_tf.SO3.from_matrix(R_c2w).wxyz
        fov_y = 2.0 * np.arctan((img_h / 2.0) / intr[1, 1])

        h = server.scene.add_camera_frustum(
            f"/scene/cameras/cam_{i}",
            fov=fov_y,
            aspect=img_w / img_h,
            scale=0.05,
            thickness=0.002,
            color=(140, 140, 140),
            wxyz=q_wxyz,
            position=cam_pos,
            visible=False,
        )
        cam_handles.append(h)

    # Step 5: Add Clean High-Confidence 3D Object Bounding Boxes and Labels
    object_labels = [o.label for o in clean_objects]
    unique_labels = sorted(list(set(object_labels)))

    color_palette = [
        (0, 200, 255), (255, 120, 0), (255, 0, 180),
        (0, 255, 120), (255, 220, 0), (160, 100, 255),
        (255, 70, 70), (70, 150, 255)
    ]

    for idx, obj in enumerate(clean_objects):
        c = obj.centroid_3d
        dims = np.maximum(obj.bbox_3d_max - obj.bbox_3d_min, 0.15)
        box_color = color_palette[idx % len(color_palette)]

        server.scene.add_box(
            f"/objects/box_{idx}_{obj.label}",
            dimensions=dims,
            position=c,
            color=box_color,
            opacity=0.28,
            wireframe=True,
        )
        server.scene.add_label(
            f"/objects/label_{idx}_{obj.label}",
            text=f"{obj.label} ({obj.confidence:.2f})",
            position=c + np.array([0, 0, dims[2] / 2.0 + 0.08]),
        )

    # Permanent 3D Start Beacon (Green) at the Fixed Room Entrance
    fixed_start_3d = navigator.fixed_start_3d
    server.scene.add_point_cloud(
        "/scene/permanent_start_marker",
        points=np.array([fixed_start_3d]),
        colors=np.array([[0, 230, 0]], dtype=np.uint8),
        point_size=0.18,
    )
    server.scene.add_label(
        "/scene/permanent_start_label",
        text="🟢 DÉPART FIXE (ENTRÉE PIÈCE)",
        position=fixed_start_3d + np.array([0, 0, 0.28]),
    )

    # Step 6: Setup Fixed Top-Left Minimap HUD & GUI Controls
    initial_map = navigator._render_map([], [], navigator.fixed_start_cell, (0, 0), None, target_width=650)

    # 1. FIXED TOP-LEFT MINIMAP (HTML Overlay)
    hud_minimap = server.gui.add_html(make_minimap_html(initial_map))

    # 2. RIGHT CONTROL SIDEBAR
    server.gui.add_markdown("# 🧭 Navigateur 3D Immersif")

    # Render Mode Dropdown (Point Cloud vs 3DGS)
    if splat_handle is not None:
        render_mode = server.gui.add_dropdown(
            "Mode d'affichage 3D :",
            options=["⚡ Rendu 3D Net Haute Définition (Recommandé)", "☁️ 3D Gaussian Splats (Expérimental)"],
            initial_value="⚡ Rendu 3D Net Haute Définition (Recommandé)",
        )

    # Point Density / Count Slider
    slider_density = server.gui.add_slider(
        "Nombre de points 3D (Résolution)",
        min=100000,
        max=min(1200000, total_pts),
        step=50000,
        initial_value=default_num_points,
    )

    # Point size slider
    slider_size = server.gui.add_slider(
        "Taille des points",
        min=0.006,
        max=0.035,
        step=0.002,
        initial_value=default_point_size,
    )

    if splat_handle is not None:
        def on_render_mode_change(_):
            is_splats = "Gaussian" in render_mode.value
            splat_handle.visible = is_splats
            pcd_handle.visible = not is_splats

        render_mode.on_update(on_render_mode_change)

    def on_density_change(_):
        n = int(slider_density.value)
        s = max(1, total_pts // n)
        pcd_handle.points = pcd_pts[::s]
        pcd_handle.colors = pcd_cols_uint8[::s]

    def on_size_change(_):
        pcd_handle.point_size = float(slider_size.value)

    slider_density.on_update(on_density_change)
    slider_size.on_update(on_size_change)

    # Camera frustum display toggle
    toggle_cams = server.gui.add_checkbox("📷 Afficher les plans de caméras", initial_value=False)

    def on_toggle_cams(_):
        for h in cam_handles:
            h.visible = toggle_cams.value

    toggle_cams.on_update(on_toggle_cams)

    default_obj = unique_labels[0] if unique_labels else "bed"
    if unique_labels:
        dropdown = server.gui.add_dropdown(
            "1. Choisir un objet détecté :",
            options=unique_labels,
            initial_value=default_obj,
        )
    else:
        dropdown = None

    text_input = server.gui.add_text("2. Ou taper un nom d'objet libre :", initial_value=default_obj)

    # Sync dropdown selection with text input
    if dropdown:
        def on_dropdown_change(_):
            text_input.value = dropdown.value
        dropdown.on_update(on_dropdown_change)

    btn_plan = server.gui.add_button("🚀 Calculer le chemin vers l'objet", color="green")

    # Quick buttons for detected high-confidence objects
    server.gui.add_markdown("### ⚡ Objets disponibles (clic direct) :")
    quick_btns = {}
    for lbl in unique_labels:
        quick_btns[lbl] = server.gui.add_button(f"🎯 Aller vers: {lbl}")

    status_msg = server.gui.add_markdown(
        f"**Statut :** Prêt. Départ fixe à l'entrée : `{np.round(fixed_start_3d, 2)}`."
    )

    last_navigation_result: Optional[NavigationResult] = None

    # Path planning execution function
    def execute_navigation(query: str):
        nonlocal last_navigation_result
        query = query.strip()
        if not query:
            status_msg.content = "⚠️ *Veuillez spécifier un nom d'objet.*"
            return

        status_msg.content = f"Statut: *Recherche et calcul de trajectoire pour '{query}'...*"
        print(f"[App] Executing navigation from fixed entrance to: '{query}'")

        # Always start from the fixed room entrance!
        result: Optional[NavigationResult] = navigator.navigate_to_object(query, start_world_pos=fixed_start_3d)
        if result is None:
            available = ", ".join(unique_labels)
            status_msg.content = (
                f"❌ **Objet '{query}' non trouvé ou inaccessible.**\n\n"
                f"Objets disponibles : *{available}*"
            )
            return

        last_navigation_result = result
        target = result.target_object
        direct_dist = float(np.linalg.norm(result.goal_world_3d - fixed_start_3d))
        status_msg.content = (
            f"✅ **Chemin trouvé vers : `{target.label}`** (confiance: {target.confidence:.2f})\n\n"
            f"- **Point de départ :** Fixe à l'entrée `({fixed_start_3d[0]:.2f}, {fixed_start_3d[1]:.2f}, {fixed_start_3d[2]:.2f})`\n"
            f"- **Distance directe :** `{direct_dist:.2f} m`\n"
            f"- **Longueur trajectoire A* :** `{len(result.path_grid)}` pas\n"
            f"- **Waypoints lissés :** `{len(result.smoothed_path_3d)}` points\n"
            f"- **Position 3D cible :** `({target.centroid_3d[0]:.2f}, {target.centroid_3d[1]:.2f}, {target.centroid_3d[2]:.2f})`"
        )

        # Update Fixed Top-Left Minimap
        hud_minimap.content = make_minimap_html(result.occupancy_map_image)

        # Clear previous trajectory & target marker in 3D scene
        for name in [
            "/navigation/path",
            "/navigation/waypoints",
            "/navigation/target_marker",
            "/navigation/target_label",
        ]:
            try:
                server.scene.remove_by_name(name)
            except Exception:
                pass

        # Draw thick 3D trajectory spline in the room
        waypoints = result.smoothed_path_3d
        if len(waypoints) >= 2:
            server.scene.add_spline_catmull_rom(
                "/navigation/path",
                positions=waypoints,
                color=(255, 30, 30),
                line_width=14.0,
                tension=0.5,
            )

            # Draw glowing sphere waypoints along the route
            server.scene.add_point_cloud(
                "/navigation/waypoints",
                points=waypoints,
                colors=np.tile([255, 80, 80], (len(waypoints), 1)),
                point_size=0.07,
                point_shape="circle",
            )

        # Target marker (Orange sphere + label)
        server.scene.add_point_cloud(
            "/navigation/target_marker",
            points=np.array([result.goal_world_3d]),
            colors=np.array([[255, 140, 0]], dtype=np.uint8),
            point_size=0.18,
        )
        server.scene.add_label(
            "/navigation/target_label",
            text=f"🎯 CIBLE : {target.label.upper()}",
            position=result.goal_world_3d + np.array([0, 0, 0.35]),
        )

    # Connect main button
    def on_main_btn_click(_):
        target_name = text_input.value.strip() or (dropdown.value if dropdown else "")
        execute_navigation(target_name)

    btn_plan.on_click(on_main_btn_click)

    # Connect quick buttons
    for lbl, btn in quick_btns.items():
        def make_quick_handler(label):
            def handler(_):
                text_input.value = label
                if dropdown:
                    dropdown.value = label
                execute_navigation(label)
            return handler
        btn.on_click(make_quick_handler(lbl))

    # -------------------------------------------------------------
    # 🎮 FIRST-PERSON GAMEPLAY / WALK-THROUGH MODE
    # -------------------------------------------------------------
    server.gui.add_markdown("---")
    server.gui.add_markdown("## 🎮 Mode Jeu Vidéo (Vue 1ère Personne)")

    # Upward vector in world frame (computed from ground plane / rotation matrix)
    if navigator.grid_data is not None and "rotation_matrix" in navigator.grid_data:
        r_inv = np.linalg.inv(navigator.grid_data["rotation_matrix"])
        up_norm = r_inv @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up_norm = up_norm / np.linalg.norm(up_norm)
    else:
        up_norm = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    room_center = np.mean(pcd_pts, axis=0)
    eye_height_offset = up_norm * 0.45  # Eye-level above floor in world units

    btn_enter_fpv = server.gui.add_button("🎮 Entrer dans la pièce (Vue 1ère Personne)", color="blue")
    btn_walk_path = server.gui.add_button("🚶 Marcher le long du chemin A* (Animé)", color="green")

    # Step size slider
    slider_step_size = server.gui.add_slider(
        "Pas de déplacement",
        min=0.05,
        max=0.40,
        step=0.05,
        initial_value=0.18,
    )

    # Directional buttons like a gamepad
    server.gui.add_markdown("**🕹️ Commandes de déplacement :**")
    bg_fwd = server.gui.add_button_group("Avant / Arrière", ["⬆️ Avancer", "⬇️ Reculer"])
    bg_strafe = server.gui.add_button_group("Pas latéraux", ["⬅️ Pas Gauche", "➡️ Pas Droite"])
    bg_turn = server.gui.add_button_group("Regard (Rotation)", ["🔄 Pivoter Gauche (15°)", "🔃 Pivoter Droite (15°)"])
    bg_vert = server.gui.add_button_group("Hauteur", ["⏫ Monter (Haut)", "⏬ Descendre (Bas)"])
    btn_overview = server.gui.add_button("🌐 Vue globale (Sortir)")

    def rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1.0 - cos_a)

    def move_clients_camera(direction: str):
        step_len = float(slider_step_size.value)
        for client in server.get_clients().values():
            pos = np.array(client.camera.position, dtype=np.float64)
            look = np.array(client.camera.look_at, dtype=np.float64)
            view_vec = look - pos
            dist = np.linalg.norm(view_vec)
            if dist < 1e-4:
                dist = 1.0
                view_vec = np.array([0.0, 0.0, 1.0])

            fwd = view_vec / dist
            # Horizontal forward (perpendicular to up)
            proj_up = np.dot(fwd, up_norm) * up_norm
            fwd_horiz = fwd - proj_up
            if np.linalg.norm(fwd_horiz) > 1e-4:
                fwd_horiz = fwd_horiz / np.linalg.norm(fwd_horiz)
            else:
                fwd_horiz = fwd

            # Right vector
            right = np.cross(fwd_horiz, up_norm)
            if np.linalg.norm(right) > 1e-4:
                right = right / np.linalg.norm(right)

            if direction == "forward":
                delta = fwd_horiz * step_len
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "backward":
                delta = -fwd_horiz * step_len
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "left":
                delta = -right * step_len
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "right":
                delta = right * step_len
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "up":
                delta = up_norm * (step_len * 0.7)
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "down":
                delta = -up_norm * (step_len * 0.7)
                client.camera.position = pos + delta
                client.camera.look_at = look + delta
            elif direction == "turn_left":
                new_view = rotate_vector_around_axis(view_vec, up_norm, 0.26)
                client.camera.look_at = pos + new_view
            elif direction == "turn_right":
                new_view = rotate_vector_around_axis(view_vec, up_norm, -0.26)
                client.camera.look_at = pos + new_view

    # Connect movement buttons
    def on_fwd_click(e):
        if "Avancer" in e.target.value:
            move_clients_camera("forward")
        else:
            move_clients_camera("backward")
    bg_fwd.on_click(on_fwd_click)

    def on_strafe_click(e):
        if "Gauche" in e.target.value:
            move_clients_camera("left")
        else:
            move_clients_camera("right")
    bg_strafe.on_click(on_strafe_click)

    def on_turn_click(e):
        if "Gauche" in e.target.value:
            move_clients_camera("turn_left")
        else:
            move_clients_camera("turn_right")
    bg_turn.on_click(on_turn_click)

    def on_vert_click(e):
        if "Monter" in e.target.value:
            move_clients_camera("up")
        else:
            move_clients_camera("down")
    bg_vert.on_click(on_vert_click)

    # Enter FPV mode
    def on_enter_fpv(_):
        eye_pos = fixed_start_3d + eye_height_offset
        look_target = room_center + up_norm * 0.10
        for client in server.get_clients().values():
            client.camera.up_direction = up_norm
            client.camera.position = eye_pos
            client.camera.look_at = look_target
    btn_enter_fpv.on_click(on_enter_fpv)

    # Global overview
    def on_overview(_):
        overview_pos = room_center + up_norm * 1.8 - np.array([0.0, 0.0, 1.2])
        for client in server.get_clients().values():
            client.camera.up_direction = up_norm
            client.camera.position = overview_pos
            client.camera.look_at = room_center
    btn_overview.on_click(on_overview)

    # Animated walkthrough along A* path
    walk_thread_running = False

    def on_walk_path(_):
        nonlocal walk_thread_running
        if walk_thread_running:
            return
        if last_navigation_result is None or len(last_navigation_result.smoothed_path_3d) < 2:
            status_msg.content = "⚠️ *Calculez d'abord un chemin vers un objet avant de marcher !*"
            return

        def run_walk():
            nonlocal walk_thread_running
            walk_thread_running = True
            try:
                waypoints = last_navigation_result.smoothed_path_3d
                # Sub-sample/interpolate waypoints into smooth steps of ~0.04m
                dense_steps = []
                for i in range(len(waypoints) - 1):
                    p0 = waypoints[i]
                    p1 = waypoints[i + 1]
                    seg_len = np.linalg.norm(p1 - p0)
                    n_sub = max(1, int(seg_len / 0.035))
                    for t in np.linspace(0, 1, n_sub, endpoint=False):
                        dense_steps.append((1.0 - t) * p0 + t * p1)
                dense_steps.append(waypoints[-1])

                for i in range(len(dense_steps)):
                    curr = dense_steps[i] + eye_height_offset
                    look_idx = min(i + 4, len(dense_steps) - 1)
                    if look_idx != i:
                        target_look = dense_steps[look_idx] + eye_height_offset
                    else:
                        target_look = last_navigation_result.goal_world_3d + eye_height_offset

                    for client in server.get_clients().values():
                        client.camera.up_direction = up_norm
                        client.camera.position = curr
                        client.camera.look_at = target_look
                    time.sleep(0.04)
            finally:
                walk_thread_running = False

        threading.Thread(target=run_walk, daemon=True).start()

    btn_walk_path.on_click(on_walk_path)

    # Trigger initial navigation to default object so path is visible immediately
    if default_obj:
        print(f"[App] Initializing default path for '{default_obj}'...")
        execute_navigation(default_obj)

    print("\n" + "=" * 60)
    print(f"  Visualiseur 3D prêt ! Ouvrez http://localhost:{port} dans votre navigateur.")
    print("=" * 60)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Scene Object Navigation with 3DGS")
    parser.add_argument(
        "--images",
        type=str,
        default=str(ROOT_DIR / "data" / "custom_frames"),
        help="Path to folder containing multi-view images",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--no-segment", action="store_true", help="Skip SAM object segmentation")
    parser.add_argument(
        "--classes",
        type=str,
        default="bed, chair, table, desk, sofa, pillow, lamp, laptop, monitor, backpack, bottle, door, window",
        help="Comma-separated target object classes",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.50,
        help="Minimum confidence threshold for displaying 3D objects",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/scene_cache.pt",
        help="Path to precomputed scene cache file (.pt)",
    )
    args = parser.parse_args()

    run_app(
        image_dir=args.images,
        port=args.port,
        run_segmentation=not args.no_segment,
        classes_str=args.classes,
        min_confidence=args.min_conf,
        cache_path=args.cache,
    )


if __name__ == "__main__":
    main()
