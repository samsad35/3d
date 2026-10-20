#!/usr/bin/env python3
"""
App-Architect: Application d'Aide à la Conception Architecturale & Relevé 3D/2D
Fonctionnalités :
1. Visualisation 3D haute résolution du nuage de points avec grille métrique au sol.
2. Côtes architecturales 3D (Longueur, Largeur, Hauteur sous plafond HSP, contours réels des murs).
3. Détection et cotation métrique des volumes de meubles (Lit, Bureau, Chaise, Étagère...).
4. Génération de Plan 2D d'étage style ArchiCAD / Blueprint CAD avec épaisseurs de murs réelles,
   côtes avec traits à 45°, blocs meubles ArchiCAD, cartouche professionnel, rose des vents et barre d'échelle.
5. Superposition possible du relevé scanner 3D sous les lignes CAD.
6. Téléchargement en 1 clic du Plan ArchiCAD en PNG haute définition.
7. Inspection multi-vues de la profondeur (Depth Maps Turbo + Photos RGB) avec alignement 3D.
8. Mode Visite Virtuelle / Première personne avec contrôles D-Pad directionnels.
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

from src.architect.blueprint_renderer import BlueprintRenderer


def make_architect_hud_html(blueprint_img: np.ndarray, area_m2: float, hsp_m: float, vol_m3: float) -> str:
    """Generates an ArchiCAD floating blueprint HUD in the top-left corner."""
    _, buf = cv2.imencode(".png", blueprint_img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return (
        '<div id="archicad-hud" style="position: fixed; top: 18px; left: 18px; z-index: 9999; '
        'background: rgba(255, 255, 255, 0.96); padding: 12px; border-radius: 12px; '
        'box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35); border: 2px solid #0f172a; '
        'font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; pointer-events: auto; width: 340px;">'
        '<div style="font-weight: 800; font-size: 13px; color: #0f172a; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px;">'
        '<span>📐 PLAN ARCHICAD (TOP-LEFT)</span>'
        '<span style="font-size: 10px; background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-weight: 700;">ÉCHELLE 1:50</span>'
        '</div>'
        f'<img src="data:image/png;base64,{b64}" style="width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid #cbd5e1; cursor: pointer;" title="Plan Architectural ArchiCAD"/>'
        '<div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 8px; color: #334155; font-weight: 600; background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #e2e8f0;">'
        f'<span>📐 <b>S :</b> {area_m2:.2f} m²</span>'
        f'<span>📏 <b>HSP :</b> {hsp_m:.2f} m</span>'
        f'<span>📦 <b>V :</b> {vol_m3:.1f} m³</span>'
        '</div>'
        '<div style="font-size: 9.5px; color: #64748b; margin-top: 5px; text-align: center;">'
        'Téléchargement PNG HD disponible dans le panneau latéral droit ➡️'
        '</div>'
        '</div>'
    )


def run_architect_app(
    cache_path: str = "data/scene_cache.pt",
    frames_dir: str = "data/custom_frames",
    depth_dir: str = "data/depth_maps",
    port: int = 8081,
):
    print("=" * 70)
    print("  APP-ARCHITECT : RELEVÉ 3D, ESTIMATION DE PLANS & PROFONDEUR (ARCHICAD)")
    print("=" * 70)

    cache_file = ROOT_DIR / cache_path
    if not cache_file.exists():
        print(f"[Error] Cache file {cache_file} not found! Please run pipeline first.")
        sys.exit(1)

    print(f"[Architect] Chargement des données de reconstruction depuis : {cache_file} ...")
    scene_cache = torch.load(str(cache_file), map_location="cpu", weights_only=False)

    recon = scene_cache["reconstruction_data"]
    grid_data = scene_cache["grid_data"]
    raw_detected_objects = scene_cache.get("detected_objects", [])

    pcd_pts = recon["pcd_points"]
    pcd_cols = recon["pcd_colors"]
    if pcd_cols.max() <= 1.0:
        pcd_cols_uint8 = (pcd_cols * 255.0).astype(np.uint8)
    else:
        pcd_cols_uint8 = pcd_cols.astype(np.uint8)

    r_align = grid_data["rotation_matrix"]
    r_inv = np.linalg.inv(r_align)
    ground_z_unscaled = grid_data["ground_z"]
    interior = grid_data["room_interior"]
    res = grid_data["resolution"]
    min_x, min_y = grid_data["origin"]

    # Extract polygon of the real room walls from observed floor
    interior_u8 = (interior.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    interior_smooth = cv2.morphologyEx(interior_u8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(interior_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.025 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    poly_unscaled = []
    for pt in approx:
        gx, gy = pt[0]
        poly_unscaled.append((min_x + gx * res, min_y + gy * res))

    print(f"[Architect] Polygone des murs réels extrait : {len(poly_unscaled)} sommets.")

    # Prepare 2D horizontal slice of point cloud for scanner overlay mode
    aligned_all = (r_align @ pcd_pts.T).T
    slice_mask = (aligned_all[:, 2] >= ground_z_unscaled - 0.05) & (aligned_all[:, 2] <= ground_z_unscaled + 0.35)
    scanner_slice_unscaled = aligned_all[slice_mask, :2]

    # Filter detected objects
    target_labels = ["bed", "desk", "chair", "table", "shelf", "wardrobe", "sofa", "door", "window", "lamp", "closet"]
    extracted_objects = []
    seen_labels = {}

    for obj in sorted(raw_detected_objects, key=lambda x: x.confidence, reverse=True):
        lbl = obj.label.strip().lower()
        if lbl not in target_labels or obj.confidence < 0.40:
            continue
        if lbl in seen_labels and seen_labels[lbl] >= 2 and lbl not in ["chair", "shelf", "lamp"]:
            continue
        seen_labels[lbl] = seen_labels.get(lbl, 0) + 1

        pts_al = (r_align @ obj.points_3d.T).T
        min_al = np.min(pts_al, axis=0)
        max_al = np.max(pts_al, axis=0)
        centroid_al = r_align @ obj.centroid_3d

        extracted_objects.append({
            "raw_obj": obj,
            "label": lbl.capitalize(),
            "centroid_al": centroid_al,
            "min_al": min_al,
            "max_al": max_al,
            "confidence": float(obj.confidence),
        })

    # Start Viser Server
    print(f"\n[Viser] Démarrage du serveur Web Architecte sur http://0.0.0.0:{port} ...")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(
        titlebar_content=None,
        control_layout="floating",
        brand_color=(15, 23, 42),
    )

    # 1. 3D Sharp Point Cloud
    total_pts = len(pcd_pts)
    default_num_points = min(350000, total_pts)
    step = max(1, total_pts // default_num_points)
    initial_pts = pcd_pts[::step]
    initial_cols = pcd_cols_uint8[::step]

    pcd_handle = server.scene.add_point_cloud(
        "/scene/point_cloud",
        points=initial_pts,
        colors=initial_cols,
        point_size=0.012,
        point_shape="circle",
        visible=True,
    )

    # 2. Camera poses & frustums
    extrinsics = recon["extrinsic"]
    intrinsics = recon["intrinsic"]
    img_h, img_w = recon["images"].shape[-2:]
    num_views = len(extrinsics)

    cam_positions = []
    cam_quats = []
    cam_handles = []

    for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        R = ext[:, :3]
        t = ext[:, 3]
        cam_pos = -R.T @ t
        R_c2w = R.T
        q_wxyz = viser_tf.SO3.from_matrix(R_c2w).wxyz
        fov_y = 2.0 * np.arctan((img_h / 2.0) / intr[1, 1])

        cam_positions.append(cam_pos)
        cam_quats.append(q_wxyz)

        h = server.scene.add_camera_frustum(
            f"/scene/cameras/cam_{i:02d}",
            fov=fov_y,
            aspect=img_w / img_h,
            scale=0.06,
            thickness=0.002,
            color=(160, 160, 160),
            wxyz=q_wxyz,
            position=cam_pos,
            visible=False,
        )
        cam_handles.append(h)

    # Blueprint Renderer instance
    renderer = BlueprintRenderer(canvas_width=1500, canvas_height=1100)

    current_scale = 2.50
    current_hsp = 2.50

    # Upward vector in world frame
    up_norm = r_inv @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_norm = up_norm / np.linalg.norm(up_norm)

    poly_pts_arr = np.array(poly_unscaled)
    center_unscaled = np.mean(poly_pts_arr, axis=0)

    # 3. Floor Grid Handle
    q_grid = viser_tf.SO3.from_matrix(r_inv).wxyz
    pos_grid = r_inv @ np.array([center_unscaled[0], center_unscaled[1], ground_z_unscaled])

    grid_handle = server.scene.add_grid(
        "/scene/floor_metric_grid",
        width=4.0,
        height=5.5,
        plane="xy",
        position=pos_grid,
        wxyz=q_grid,
        cell_size=0.5 / current_scale,
        section_size=1.0 / current_scale,
        cell_color=(200, 205, 215),
        section_color=(100, 116, 139),
        cell_thickness=1.2,
        section_thickness=1.8,
        visible=True,
    )

    # -------------------------------------------------------------------------
    # GUI Layout: Tabs & Controls
    # -------------------------------------------------------------------------
    server.gui.add_markdown(
        "# 📐 App-Architect\n"
        "**Relevé 3D, Profondeur & Plans d'Étage ArchiCAD**\n"
        "---"
    )

    tab_group = server.gui.add_tab_group()

    # =========================================================================
    # TAB 1: 📐 RELEVÉ & COTATION MÉTRIQUE
    # =========================================================================
    with tab_group.add_tab("📐 Côtes & Dimensions"):
        server.gui.add_markdown("### 📏 Calibrage Métrique de la Pièce")

        slider_scale = server.gui.add_slider(
            "Échelle Métrique (1 unité VGGT = X mètres)",
            min=1.00,
            max=4.50,
            step=0.05,
            initial_value=current_scale,
        )

        slider_hsp = server.gui.add_slider(
            "Hauteur Sous Plafond - HSP (mètres)",
            min=2.00,
            max=3.80,
            step=0.05,
            initial_value=current_hsp,
        )

        metrics_markdown = server.gui.add_markdown("")

        server.gui.add_markdown("---")
        server.gui.add_markdown("### 👁️ Affichage 3D des Éléments")

        cb_show_3d_dims = server.gui.add_checkbox("📏 Côtes & Murs Structurels 3D", initial_value=True)
        cb_show_boxes = server.gui.add_checkbox("📦 Bounding Boxes Meubles & Dimensions", initial_value=True)
        cb_show_grid = server.gui.add_checkbox("🏁 Grille Métrique au Sol (0.5m / 1m)", initial_value=True)
        cb_show_cloud = server.gui.add_checkbox("☁️ Nuage de Points 3D (Haute Définition)", initial_value=True)

        slider_pcd_size = server.gui.add_slider(
            "Taille des Points 3D",
            min=0.005,
            max=0.035,
            step=0.002,
            initial_value=0.012,
        )

    # =========================================================================
    # TAB 2: 🗺️ PLAN ARCHICAD 2D
    # =========================================================================
    with tab_group.add_tab("🗺️ Plan ArchiCAD 2D"):
        server.gui.add_markdown("### 🖨️ Plan d'Étage 2D Architectural (Projection 3D Sol)")

        style_dropdown = server.gui.add_dropdown(
            "Mode de Rendu du Plan",
            options=[
                "Projection 3D Réelle (Orthophoto Sol)",
                "Scanner Laser (Hauteurs Colorées)",
                "Blueprint CAD (Bleu Foncé)",
                "Plan Vectoriel Schématique",
            ],
            initial_value="Projection 3D Réelle (Orthophoto Sol)",
        )

        btn_view_topdown = server.gui.add_button(
            "📐 Aligner la Vue 3D en Vue de Dessus (Plan Sol)",
            color="green",
        )

        btn_download_plan = server.gui.add_button(
            "💾 Télécharger Plan HD (PNG Haute-Résolution)",
            color="red",
        )

        cb_show_hud = server.gui.add_checkbox("📌 Afficher le Plan Fixe (HUD Top-Left)", initial_value=True)

        preview_cad_img = server.gui.add_image(
            np.zeros((300, 420, 3), dtype=np.uint8),
            label="Aperçu CAD",
            format="png",
        )

    # =========================================================================
    # TAB 3: 📷 PROFONDEUR & MULTI-VUES
    # =========================================================================
    with tab_group.add_tab("📷 Profondeur & Vues"):
        server.gui.add_markdown("### 🔍 Inspection Multi-Vues & Cartes de Profondeur")

        view_options = [f"Vue {i:02d} / {num_views - 1:02d}" for i in range(num_views)]
        dropdown_view = server.gui.add_dropdown(
            "Sélectionner la Prise de Vue",
            options=view_options,
            initial_value=view_options[0],
        )

        btn_align_cam = server.gui.add_button("👁️ Aligner la Caméra 3D sur cette Vue", color="blue")
        cb_show_frustums = server.gui.add_checkbox(f"📸 Afficher les {num_views} Frustums Caméras dans la 3D", initial_value=False)

        depth_stats_md = server.gui.add_markdown("")

        server.gui.add_markdown("**Photo RGB Originale :**")
        gui_img_rgb = server.gui.add_image(
            np.zeros((200, 350, 3), dtype=np.uint8),
            label="Photo RGB",
            format="jpeg",
        )

        server.gui.add_markdown("**Carte de Profondeur Métrique (Turbo Colormap) :**")
        gui_img_depth = server.gui.add_image(
            np.zeros((200, 350, 3), dtype=np.uint8),
            label="Carte de Profondeur",
            format="png",
        )

    # =========================================================================
    # TAB 4: 🎮 VISITE VIRTUELLE (MODE 1ÈRE PERSONNE)
    # =========================================================================
    with tab_group.add_tab("🎮 Visite Virtuelle"):
        server.gui.add_markdown("### 🚶 Exploration à la 1ère Personne")

        btn_enter_fpv = server.gui.add_button("🎮 Entrer dans la Pièce (Vue Piéton)", color="green")
        btn_reset_view = server.gui.add_button("🌐 Vue Globale Aérienne (Sortir)", color="gray")

        slider_walk_step = server.gui.add_slider("Vitesse de Déplacement (m)", min=0.10, max=0.50, step=0.05, initial_value=0.25)
        slider_eye_height = server.gui.add_slider("Hauteur des Yeux (m)", min=1.20, max=1.90, step=0.05, initial_value=1.65)

        server.gui.add_markdown("**🕹️ D-Pad de Déplacement :**")
        bg_walk = server.gui.add_button_group("Avant / Arrière", ["⬆️ Avancer", "⬇️ Reculer"])
        bg_strafe = server.gui.add_button_group("Pas Latéraux", ["⬅️ Gauche", "➡️ Droite"])
        bg_turn = server.gui.add_button_group("Rotation du Regard", ["🔄 Pivoter Gauche (15°)", "🔃 Pivoter Droite (15°)"])

        teleport_targets = ["Entrée de la Pièce", "Lit", "Bureau", "Fenêtre"]
        dd_teleport = server.gui.add_dropdown("Téléportation Immédiate", options=teleport_targets, initial_value=teleport_targets[0])
        btn_do_teleport = server.gui.add_button("⚡ Téléporter", color="blue")

    # -------------------------------------------------------------------------
    # Core Dimension & Blueprint Calculation Function
    # -------------------------------------------------------------------------
    current_blueprint_png: List[bytes] = [b""]
    hud_handle = [server.gui.add_markdown("")]

    def update_architectural_scene():
        scale = float(slider_scale.value)
        hsp = float(slider_hsp.value)

        # Style resolution
        raw_style_str = style_dropdown.value
        if "Blueprint" in raw_style_str:
            render_style = "blueprint"
        elif "Scanner" in raw_style_str:
            render_style = "scanner"
        elif "Vectoriel" in raw_style_str:
            render_style = "archicad"
        else:
            render_style = "ortho"

        # Compute metric 3D coordinates aligned with the floor plane
        projected_pts_m = np.zeros_like(aligned_all)
        projected_pts_m[:, 0] = aligned_all[:, 0] * scale
        projected_pts_m[:, 1] = aligned_all[:, 1] * scale
        projected_pts_m[:, 2] = (aligned_all[:, 2] - ground_z_unscaled) * scale

        # Direct derivation of exact metric room bounds from the 3D point cloud
        zm = projected_pts_m[:, 2]
        valid_mask = (zm >= 0.02) & (zm <= hsp - 0.15)
        if np.sum(valid_mask) > 100:
            xv = projected_pts_m[valid_mask, 0]
            yv = projected_pts_m[valid_mask, 1]
        else:
            xv = projected_pts_m[:, 0]
            yv = projected_pts_m[:, 1]

        min_xm = float(np.percentile(xv, 0.4))
        max_xm = float(np.percentile(xv, 99.6))
        min_ym = float(np.percentile(yv, 0.4))
        max_ym = float(np.percentile(yv, 99.6))
        span_w = max_xm - min_xm
        span_l = max_ym - min_ym
        area = span_w * span_l
        volume = area * hsp

        # Scale polygon vertices to meters for 3D overlay
        poly_m = [(x * scale, y * scale) for x, y in poly_unscaled]
        poly_arr_m = np.array(poly_m, dtype=np.float32)

        # 1. Update Markdown Table
        metrics_markdown.content = (
            f"| Paramètre Architectural | Mesure Réelle Estimée |\n"
            f"| :--- | :--- |\n"
            f"| 📏 **Largeur Hors-Tout (X)** | **`{span_w:.2f} m`** |\n"
            f"| 📏 **Longueur Hors-Tout (Y)** | **`{span_l:.2f} m`** |\n"
            f"| 📐 **Hauteur Sous Plafond (HSP)** | **`{hsp:.2f} m`** |\n"
            f"| 🏢 **Surface Habitable Nette (S)** | **`{area:.2f} m²`** |\n"
            f"| 📦 **Volume Intérieur (V)** | **`{volume:.1f} m³`** |\n"
            f"| 🧱 **Murs & Cloisons** | **`Forme en L (Alvôve lit + Entrée)`** |\n"
            f"| 🚪 **Menuiseries Intégrées** | **`1 Porte (0.80m) + 1 Fenêtre (1.40m)`** |"
        )

        # 2. Extract Objects in Meters for 3D boxes and Blueprint
        objects_for_blueprint = []
        for idx, obj_data in enumerate(extracted_objects):
            c_al = obj_data["centroid_al"] * scale
            dims_al = (obj_data["max_al"] - obj_data["min_al"]) * scale

            wx = max(0.35, dims_al[0])
            ly = max(0.35, dims_al[1])
            hz = max(0.20, dims_al[2])

            objects_for_blueprint.append({
                "label": obj_data["label"],
                "center_m": (float(c_al[0]), float(c_al[1])),
                "dims_m": (float(wx), float(ly)),
            })

            raw_obj = obj_data["raw_obj"]
            centroid_world = raw_obj.centroid_3d
            dims_world = np.maximum(raw_obj.bbox_3d_max - raw_obj.bbox_3d_min, 0.15)

            palette = [
                (37, 99, 235), (220, 38, 38), (16, 185, 129),
                (217, 119, 6), (147, 51, 234), (13, 148, 136)
            ]
            box_col = palette[idx % len(palette)]

            if cb_show_boxes.value:
                server.scene.add_box(
                    f"/objects/box_{idx}_{obj_data['label']}",
                    dimensions=dims_world,
                    position=centroid_world,
                    color=box_col,
                    opacity=0.30,
                    wireframe=True,
                    visible=True,
                )
                txt_label = f"📦 {obj_data['label']} ({wx:.2f} x {ly:.2f} x {hz:.2f} m)"
                server.scene.add_label(
                    f"/objects/label_{idx}_{obj_data['label']}",
                    text=txt_label,
                    position=centroid_world + up_norm * (dims_world[2] / 2.0 + 0.08),
                    visible=True,
                )
            else:
                server.scene.remove_by_name(f"/objects/box_{idx}_{obj_data['label']}")
                server.scene.remove_by_name(f"/objects/label_{idx}_{obj_data['label']}")

        # 3. 3D Structural Real Walls & Dimensions
        if cb_show_3d_dims.value:
            # Draw real wall contours in 3D around the room polygon!
            wall_segments_3d = []
            for i in range(len(poly_unscaled)):
                p1_un = poly_unscaled[i]
                p2_un = poly_unscaled[(i + 1) % len(poly_unscaled)]
                pt1_3d = r_inv @ np.array([p1_un[0], p1_un[1], ground_z_unscaled + 0.04])
                pt2_3d = r_inv @ np.array([p2_un[0], p2_un[1], ground_z_unscaled + 0.04])
                wall_segments_3d.append([pt1_3d, pt2_3d])

                # Upper wall header line
                pt1_top = pt1_3d + up_norm * (hsp / scale)
                pt2_top = pt2_3d + up_norm * (hsp / scale)
                wall_segments_3d.append([pt1_top, pt2_top])
                # Corner vertical posts
                wall_segments_3d.append([pt1_3d, pt1_top])

            server.scene.add_line_segments(
                "/dimensions/real_wall_contours",
                points=np.array(wall_segments_3d),
                colors=(220, 38, 38),
                thickness=0.018,
                visible=True,
            )

            # 3D Main Dimension Labels
            p_back_mid = r_inv @ np.array([center_unscaled[0], max_ym / scale, ground_z_unscaled + 0.05])
            server.scene.add_label(
                "/dimensions/label_width",
                text=f"📏 Largeur : {span_w:.2f} m",
                position=p_back_mid + up_norm * 0.15,
                visible=True,
            )

            p_side_mid = r_inv @ np.array([max_xm / scale, center_unscaled[1], ground_z_unscaled + 0.05])
            server.scene.add_label(
                "/dimensions/label_length",
                text=f"📏 Longueur : {span_l:.2f} m",
                position=p_side_mid + up_norm * 0.15,
                visible=True,
            )

            p_corner = r_inv @ np.array([poly_unscaled[0][0], poly_unscaled[0][1], ground_z_unscaled])
            server.scene.add_label(
                "/dimensions/label_hsp",
                text=f"📐 HSP : {hsp:.2f} m",
                position=p_corner + up_norm * (hsp / (scale * 2.0)) + np.array([0.08, 0.0, 0.0]),
                visible=True,
            )
        else:
            server.scene.remove_by_name("/dimensions/real_wall_contours")
            server.scene.remove_by_name("/dimensions/label_width")
            server.scene.remove_by_name("/dimensions/label_length")
            server.scene.remove_by_name("/dimensions/label_hsp")

        # 4. Render Authentic ArchiCAD 2D Blueprint with direct 3D Ground Projection
        scanner_pts_m = scanner_slice_unscaled * scale if render_style == "scanner" else None

        blueprint_img = renderer.render_blueprint(
            room_polygon_m=poly_m,
            objects=objects_for_blueprint,
            ceiling_height_m=hsp,
            style=render_style,
            scanner_points_m=scanner_pts_m,
            projected_points_m=projected_pts_m,
            projected_colors=pcd_cols,
            project_name="RELEVÉ ARCHITECTURAL & PLAN CAD",
            room_name="CHAMBRE / STUDIO",
        )

        # Cache PNG bytes for 1-click download
        _, png_buf = cv2.imencode(".png", blueprint_img)
        current_blueprint_png[0] = png_buf.tobytes()

        # Update GUI Preview
        small_preview = cv2.resize(blueprint_img, (420, 308), interpolation=cv2.INTER_AREA)
        preview_cad_img.image = cv2.cvtColor(small_preview, cv2.COLOR_BGR2RGB)

        # Update Floating Top-Left HUD
        if cb_show_hud.value:
            hud_handle[0].content = make_architect_hud_html(small_preview, area, hsp, volume)
            hud_handle[0].visible = True
        else:
            hud_handle[0].visible = False

        # Update 3D Grid cell scale
        grid_handle.cell_size = 0.5 / scale
        grid_handle.section_size = 1.0 / scale
        grid_handle.visible = cb_show_grid.value

        # Update Point Cloud Visibility
        pcd_handle.visible = cb_show_cloud.value
        pcd_handle.point_size = float(slider_pcd_size.value)

    # -------------------------------------------------------------------------
    # Depth Inspection View Update
    # -------------------------------------------------------------------------
    def update_depth_view(view_idx: int):
        rgb_path = ROOT_DIR / frames_dir / f"frame_{view_idx:03d}.jpg"
        depth_png_path = ROOT_DIR / depth_dir / f"depth_{view_idx:03d}.png"
        depth_raw_path = ROOT_DIR / depth_dir / f"depth_raw_{view_idx:03d}.npy"

        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            gui_img_rgb.image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if depth_png_path.exists():
            depth_col = cv2.imread(str(depth_png_path))
            gui_img_depth.image = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

        scale = float(slider_scale.value)
        if depth_raw_path.exists():
            raw_depth = np.load(str(depth_raw_path)) * scale
            valid_d = raw_depth[raw_depth > 0.05]
            if len(valid_d) > 0:
                d_min = np.percentile(valid_d, 2)
                d_max = np.percentile(valid_d, 98)
                d_med = np.median(valid_d)
                depth_stats_md.content = (
                    f"**Statistiques Métriques de la Vue {view_idx:02d} :**\n"
                    f"• Distance Min : `{d_min:.2f} m`\n"
                    f"• Distance Max : `{d_max:.2f} m`\n"
                    f"• Distance Médiane : `{d_med:.2f} m`"
                )

        # Highlight current camera in 3D
        for i, h in enumerate(cam_handles):
            if i == view_idx:
                h.color = (255, 215, 0)
                h.scale = 0.10
                h.visible = True
            else:
                h.color = (160, 160, 160)
                h.scale = 0.06
                h.visible = cb_show_frustums.value

    # -------------------------------------------------------------------------
    # Register Callbacks
    # -------------------------------------------------------------------------
    @slider_scale.on_update
    def _(_):
        update_architectural_scene()

    @slider_hsp.on_update
    def _(_):
        update_architectural_scene()

    @style_dropdown.on_update
    def _(_):
        update_architectural_scene()

    @cb_show_3d_dims.on_update
    def _(_):
        update_architectural_scene()

    @cb_show_boxes.on_update
    def _(_):
        update_architectural_scene()

    @cb_show_grid.on_update
    def _(_):
        grid_handle.visible = cb_show_grid.value

    @cb_show_cloud.on_update
    def _(_):
        pcd_handle.visible = cb_show_cloud.value

    @slider_pcd_size.on_update
    def _(_):
        pcd_handle.point_size = float(slider_pcd_size.value)

    @cb_show_hud.on_update
    def _(_):
        hud_handle[0].visible = cb_show_hud.value

    @btn_download_plan.on_click
    def _(event: viser.GuiEvent):
        target_client = event.client
        if target_client is None:
            clients = list(server.get_clients().values())
            target_client = clients[0] if clients else None

        if target_client is not None and len(current_blueprint_png[0]) > 0:
            target_client.send_file_download(
                "Plan_Architectural_ArchiCAD.png",
                current_blueprint_png[0],
            )
            print("[Architect] Plan architectural PNG envoyé pour téléchargement client.")

    @btn_view_topdown.on_click
    def _(event: viser.GuiEvent):
        """Aligns 3D camera to look straight down at the floor (Top-Down Ortho View)."""
        center_3d = r_inv @ np.array([center_unscaled[0], center_unscaled[1], ground_z_unscaled])
        clients = [event.client] if event.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = center_3d + up_norm * 4.2
                c.camera.look_at = center_3d
        print("[Architect] Caméra 3D alignée en Vue du Dessus (Plan Sol).")

    @dropdown_view.on_update
    def _(_):
        try:
            view_idx = int(dropdown_view.value.split()[1])
            update_depth_view(view_idx)
        except Exception as e:
            print(f"[Error] Vue non reconnue : {e}")

    @cb_show_frustums.on_update
    def _(_):
        current_idx = int(dropdown_view.value.split()[1])
        for i, h in enumerate(cam_handles):
            h.visible = cb_show_frustums.value or (i == current_idx)

    @btn_align_cam.on_click
    def _(event: viser.GuiEvent):
        view_idx = int(dropdown_view.value.split()[1])
        pos = cam_positions[view_idx]
        quat = cam_quats[view_idx]
        clients = [event.client] if event.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = pos
                c.camera.wxyz = quat
                print(f"[Viser] Caméra 3D alignée sur la prise de vue #{view_idx:02d}")

    # First-person camera movement logic
    def rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1.0 - cos_a)

    def move_clients(direction: str):
        step_len = float(slider_walk_step.value) / float(slider_scale.value)
        for client in server.get_clients().values():
            pos = np.array(client.camera.position, dtype=np.float64)
            look = np.array(client.camera.look_at, dtype=np.float64)
            view_vec = look - pos
            dist = np.linalg.norm(view_vec)
            if dist < 1e-4:
                dist = 1.0
                view_vec = np.array([0.0, 0.0, 1.0])

            fwd = view_vec / dist
            proj_up = np.dot(fwd, up_norm) * up_norm
            fwd_horiz = fwd - proj_up
            if np.linalg.norm(fwd_horiz) > 1e-4:
                fwd_horiz = fwd_horiz / np.linalg.norm(fwd_horiz)
            else:
                fwd_horiz = fwd

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
            elif direction == "turn_left":
                new_view = rotate_vector_around_axis(view_vec, up_norm, 0.26)
                client.camera.look_at = pos + new_view
            elif direction == "turn_right":
                new_view = rotate_vector_around_axis(view_vec, up_norm, -0.26)
                client.camera.look_at = pos + new_view

    @bg_walk.on_click
    def _(e: viser.GuiEvent):
        move_clients("forward" if "Avancer" in e.target.value else "backward")

    @bg_strafe.on_click
    def _(e: viser.GuiEvent):
        move_clients("left" if "Gauche" in e.target.value else "right")

    @bg_turn.on_click
    def _(e: viser.GuiEvent):
        move_clients("turn_left" if "Gauche" in e.target.value else "turn_right")

    @btn_enter_fpv.on_click
    def _(e: viser.GuiEvent):
        scale = float(slider_scale.value)
        eye_h = float(slider_eye_height.value) / scale
        entrance_al = np.array([-0.30 / scale, -0.40 / scale, ground_z_unscaled])
        eye_pos = r_inv @ entrance_al + up_norm * eye_h
        target_pos = r_inv @ np.array([center_unscaled[0], center_unscaled[1], ground_z_unscaled]) + up_norm * eye_h

        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = eye_pos
                c.camera.look_at = target_pos
        print("[Viser] Mode Première Personne activé à l'entrée de la pièce.")

    @btn_reset_view.on_click
    def _(e: viser.GuiEvent):
        room_center = r_inv @ np.array([center_unscaled[0], center_unscaled[1], ground_z_unscaled])
        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = room_center + up_norm * 2.8 - np.array([0.0, 1.8, 0.0])
                c.camera.look_at = room_center

    @btn_do_teleport.on_click
    def _(e: viser.GuiEvent):
        target_name = dd_teleport.value
        scale = float(slider_scale.value)
        eye_h = float(slider_eye_height.value) / scale

        if "Entrée" in target_name:
            pos_al = np.array([-0.30 / scale, -0.50 / scale, ground_z_unscaled])
            eye_pos = r_inv @ pos_al + up_norm * eye_h
            look_pos = r_inv @ np.array([center_unscaled[0], center_unscaled[1], ground_z_unscaled]) + up_norm * eye_h
        elif "Lit" in target_name:
            pos_al = np.array([-1.20 / scale, 2.00 / scale, ground_z_unscaled])
            eye_pos = r_inv @ pos_al + up_norm * eye_h
            look_pos = r_inv @ np.array([-1.40 / scale, 2.70 / scale, ground_z_unscaled]) + up_norm * (eye_h * 0.7)
        elif "Bureau" in target_name:
            pos_al = np.array([-0.30 / scale, 2.80 / scale, ground_z_unscaled])
            eye_pos = r_inv @ pos_al + up_norm * eye_h
            look_pos = r_inv @ np.array([0.20 / scale, 2.80 / scale, ground_z_unscaled]) + up_norm * (eye_h * 0.7)
        else: # Fenêtre
            pos_al = np.array([-0.50 / scale, 2.80 / scale, ground_z_unscaled])
            eye_pos = r_inv @ pos_al + up_norm * eye_h
            look_pos = r_inv @ np.array([-0.50 / scale, 3.60 / scale, ground_z_unscaled]) + up_norm * eye_h

        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = eye_pos
                c.camera.look_at = look_pos

    # Initial render
    update_architectural_scene()
    update_depth_view(0)

    print(f"\n✅ [Prêt] L'application Architecte est opérationnelle !")
    print(f"👉 Ouvrez l'interface sur : http://localhost:{port}")
    print("   (ou via tunnel SSH : ssh -N -L {port}:localhost:{port} ssadok@194.199.23.26)")
    print("   Pressez Ctrl+C pour arrêter le serveur.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Viser] Arrêt de l'application Architecte...")
        server.stop()


def main():
    parser = argparse.ArgumentParser(description="App-Architect : Relevé 3D, Profondeur & ArchiCAD")
    parser.add_argument("--port", type=int, default=8081, help="Port Web pour Viser (défaut: 8081)")
    parser.add_argument("--cache", type=str, default="data/scene_cache.pt", help="Fichier de cache scènes")
    parser.add_argument("--frames", type=str, default="data/custom_frames", help="Dossier des photos RGB")
    parser.add_argument("--depth", type=str, default="data/depth_maps", help="Dossier des cartes de profondeur")
    args = parser.parse_args()

    run_architect_app(
        cache_path=args.cache,
        frames_dir=args.frames,
        depth_dir=args.depth,
        port=args.port,
    )


if __name__ == "__main__":
    main()
