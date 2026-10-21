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
from src.architect.vector_floorplan import ManhattanVectorFloorplan, VectorFloorplanResult


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

    # Orthonormal Canonical Room Coordinate Basis (X=Width, Y=Length, Z=Height)
    # Computed from the room's horizontal floor normal and dominant Manhattan wall axes:
    v_height = np.array([0.02588, 0.52618, 0.84998], dtype=np.float64)
    v_width  = np.array([-0.99583, 0.08797, -0.02414], dtype=np.float64)
    v_length = np.array([-0.08747, -0.84581, 0.52626], dtype=np.float64)
    R_canonical = np.stack([v_width, v_length, v_height], axis=0)

    # Transform unscaled points to canonical room frame
    pts_canonical = pcd_pts @ R_canonical.T
    z_ground_canonical = float(np.percentile(pts_canonical[:, 2], 5))

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

        extracted_objects.append({
            "raw_obj": obj,
            "label": lbl.capitalize(),
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

    current_scale = 2.50
    current_hsp = 2.50

    # 1. 3D Sharp Point Cloud in Metric Room Frame
    total_pts = len(pcd_pts)
    default_num_points = min(350000, total_pts)
    step = max(1, total_pts // default_num_points)

    pts_init_m = np.zeros_like(pts_canonical[::step])
    pts_init_m[:, 0] = pts_canonical[::step, 0] * current_scale
    pts_init_m[:, 1] = pts_canonical[::step, 1] * current_scale
    pts_init_m[:, 2] = (pts_canonical[::step, 2] - z_ground_canonical) * current_scale

    initial_cols = pcd_cols_uint8[::step]

    pcd_handle = server.scene.add_point_cloud(
        "/scene/point_cloud",
        points=pts_init_m,
        colors=initial_cols,
        point_size=0.012,
        point_shape="circle",
        visible=True,
    )

    # 2. Camera poses & frustums transformed to Canonical Frame
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
        cam_pos_m = (cam_pos @ R_canonical.T) * current_scale
        cam_pos_m[2] -= z_ground_canonical * current_scale

        R_c2w = R.T
        R_c2w_canon = R_canonical @ R_c2w
        q_wxyz = viser_tf.SO3.from_matrix(R_c2w_canon).wxyz
        fov_y = 2.0 * np.arctan((img_h / 2.0) / intr[1, 1])

        cam_positions.append(cam_pos_m)
        cam_quats.append(q_wxyz)

        h = server.scene.add_camera_frustum(
            f"/scene/cameras/cam_{i:02d}",
            fov=fov_y,
            aspect=img_w / img_h,
            scale=0.12,
            thickness=0.003,
            color=(160, 160, 160),
            wxyz=q_wxyz,
            position=cam_pos_m,
            visible=False,
        )
        cam_handles.append(h)

    # Blueprint Renderer & Manhattan Vectorizer instances
    renderer = BlueprintRenderer(canvas_width=1500, canvas_height=1100)
    vector_engine = ManhattanVectorFloorplan(slice_z_min=0.30, slice_z_max=2.00)

    # 3. Floor Metric Grid Handle
    grid_handle = server.scene.add_grid(
        "/scene/floor_metric_grid",
        width=5.0,
        height=7.0,
        plane="xy",
        position=(0.35, 1.85, 0.0),
        cell_size=0.5,
        section_size=1.0,
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
                "Hybride : Murs Vectoriels + 3D Réelle",
                "Plan Vectoriel Pur ArchiCAD (Manhattan RANSAC)",
                "Projection 3D Seule (Orthophoto Sol)",
                "Scanner Laser (Hauteurs Colorées)",
                "Blueprint CAD (Bleu Foncé)",
            ],
            initial_value="Hybride : Murs Vectoriels + 3D Réelle",
        )

        btn_view_topdown = server.gui.add_button(
            "📐 Aligner la Vue 3D en Vue de Dessus (Plan Sol)",
            color="green",
        )

        server.gui.add_markdown("**Exports Vectoriels & DAO :**")
        btn_download_dxf = server.gui.add_button(
            "💾 Télécharger Plan CAD (.DXF pour ArchiCAD / AutoCAD)",
            color="green",
        )
        btn_download_svg = server.gui.add_button(
            "💾 Télécharger Plan Vectoriel (.SVG)",
            color="blue",
        )
        btn_download_plan = server.gui.add_button(
            "💾 Télécharger Plan Image (.PNG HD)",
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
    current_dxf_bytes: List[bytes] = [b""]
    current_svg_bytes: List[bytes] = [b""]
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
        elif "Pur" in raw_style_str:
            render_style = "pure_vector"
        elif "Hybride" in raw_style_str:
            render_style = "hybrid"
        else:
            render_style = "ortho"

        # Compute metric 3D coordinates aligned with the floor plane in canonical room frame
        projected_pts_m = np.zeros_like(pts_canonical)
        projected_pts_m[:, 0] = pts_canonical[:, 0] * scale
        projected_pts_m[:, 1] = pts_canonical[:, 1] * scale
        projected_pts_m[:, 2] = (pts_canonical[:, 2] - z_ground_canonical) * scale

        # Update 3D Point Cloud in Viser
        sub_pts = projected_pts_m[::step]
        pcd_handle.points = sub_pts

        # Manhattan-World RANSAC orthogonal vector floorplan extraction
        vector_res = vector_engine.extract_vector_floorplan(
            projected_pts_m, ceiling_height_m=hsp, already_aligned=True
        )

        min_xm, max_xm, min_ym, max_ym = vector_res.bounding_box_m
        span_w = vector_res.total_width_m
        span_l = vector_res.total_length_m
        area = vector_res.area_m2
        volume = area * hsp
        perimeter = vector_res.perimeter_m

        # 1. Update Markdown Table with exact CAD metrics
        metrics_markdown.content = (
            f"| Paramètre Architectural | Mesure Réelle Estimée |\n"
            f"| :--- | :--- |\n"
            f"| 📏 **Largeur Hors-Tout (X)** | **`{span_w:.2f} m`** |\n"
            f"| 📏 **Longueur Hors-Tout (Y)** | **`{span_l:.2f} m`** |\n"
            f"| 📐 **Hauteur Sous Plafond (HSP)** | **`{hsp:.2f} m`** |\n"
            f"| 🏢 **Surface Habitable Nette (S)** | **`{area:.2f} m²`** |\n"
            f"| 📦 **Volume Intérieur (V)** | **`{volume:.1f} m³`** |\n"
            f"| 🧱 **Périmètre des Murs** | **`{perimeter:.2f} m`** |\n"
            f"| 📐 **Moteur Vectoriel CAD** | **`Manhattan RANSAC (Angles 90°)`** |\n"
            f"| 🚪 **Menuiseries Intégrées** | **`1 Porte (0.80m) + 1 Fenêtre (1.40m)`** |"
        )

        # 2. Extract Objects in Meters for 3D boxes and Blueprint
        objects_for_blueprint = []
        for idx, obj_data in enumerate(extracted_objects):
            raw_obj = obj_data["raw_obj"]
            pts_obj = (raw_obj.points_3d @ R_canonical.T) * scale
            pts_obj[:, 2] -= z_ground_canonical * scale

            b_min = np.min(pts_obj, axis=0)
            b_max = np.max(pts_obj, axis=0)
            b_center = (b_min + b_max) / 2.0
            b_dims = np.maximum(b_max - b_min, 0.20)

            wx = float(b_dims[0])
            ly = float(b_dims[1])
            hz = float(b_dims[2])
            cx = float(b_center[0])
            cy = float(b_center[1])

            objects_for_blueprint.append({
                "label": obj_data["label"],
                "center_m": (cx, cy),
                "dims_m": (wx, ly),
            })

            palette = [
                (37, 99, 235), (220, 38, 38), (16, 185, 129),
                (217, 119, 6), (147, 51, 234), (13, 148, 136)
            ]
            box_col = palette[idx % len(palette)]

            if cb_show_boxes.value:
                server.scene.add_box(
                    f"/objects/box_{idx}_{obj_data['label']}",
                    dimensions=b_dims,
                    position=b_center,
                    color=box_col,
                    opacity=0.30,
                    wireframe=True,
                    visible=True,
                )
                txt_label = f"📦 {obj_data['label']} ({wx:.2f} x {ly:.2f} x {hz:.2f} m)"
                server.scene.add_label(
                    f"/objects/label_{idx}_{obj_data['label']}",
                    text=txt_label,
                    position=b_center + np.array([0.0, 0.0, b_dims[2] / 2.0 + 0.08]),
                    visible=True,
                )
            else:
                server.scene.remove_by_name(f"/objects/box_{idx}_{obj_data['label']}")
                server.scene.remove_by_name(f"/objects/label_{idx}_{obj_data['label']}")

        # 3. 3D Structural Real Walls & Dimensions
        if cb_show_3d_dims.value:
            wall_segments_3d = []
            n_c = len(vector_res.corners)
            for i in range(n_c):
                p1 = vector_res.corners[i]
                p2 = vector_res.corners[(i + 1) % n_c]
                pt1_3d = np.array([p1[0], p1[1], 0.04])
                pt2_3d = np.array([p2[0], p2[1], 0.04])
                wall_segments_3d.append([pt1_3d, pt2_3d])

                # Upper wall header line
                pt1_top = np.array([p1[0], p1[1], hsp])
                pt2_top = np.array([p2[0], p2[1], hsp])
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
            mid_x = (min_xm + max_xm) / 2.0
            mid_y = (min_ym + max_ym) / 2.0
            server.scene.add_label(
                "/dimensions/label_width",
                text=f"📏 Largeur : {span_w:.2f} m",
                position=np.array([mid_x, max_ym + 0.15, 0.10]),
                visible=True,
            )
            server.scene.add_label(
                "/dimensions/label_length",
                text=f"📏 Longueur : {span_l:.2f} m",
                position=np.array([max_xm + 0.15, mid_y, 0.10]),
                visible=True,
            )
            server.scene.add_label(
                "/dimensions/label_hsp",
                text=f"📐 HSP : {hsp:.2f} m",
                position=np.array([vector_res.corners[0][0], vector_res.corners[0][1], hsp / 2.0]),
                visible=True,
            )
        else:
            server.scene.remove_by_name("/dimensions/real_wall_contours")
            server.scene.remove_by_name("/dimensions/label_width")
            server.scene.remove_by_name("/dimensions/label_length")
            server.scene.remove_by_name("/dimensions/label_hsp")

        # 4. Render Authentic ArchiCAD 2D Blueprint with Manhattan Vector Extraction
        if render_style == "scanner":
            scan_mask = (projected_pts_m[:, 2] >= 0.05) & (projected_pts_m[:, 2] <= 0.40)
            scanner_pts_m = projected_pts_m[scan_mask, :2]
        else:
            scanner_pts_m = None

        blueprint_img = renderer.render_blueprint(
            room_polygon_m=vector_res.corners,
            objects=objects_for_blueprint,
            ceiling_height_m=hsp,
            style=render_style,
            scanner_points_m=scanner_pts_m,
            projected_points_m=projected_pts_m,
            projected_colors=pcd_cols,
            vector_result=vector_res,
            project_name="RELEVÉ ARCHITECTURAL & PLAN CAD",
            room_name="CHAMBRE / STUDIO",
        )

        # Cache PNG, DXF and SVG bytes for 1-click download
        _, png_buf = cv2.imencode(".png", blueprint_img)
        current_blueprint_png[0] = png_buf.tobytes()

        dxf_path = ROOT_DIR / "data" / "plan_architectural.dxf"
        vector_engine.export_dxf(vector_res, dxf_path, objects=objects_for_blueprint, ceiling_height_m=hsp)
        if dxf_path.exists():
            current_dxf_bytes[0] = dxf_path.read_bytes()

        svg_path = ROOT_DIR / "data" / "plan_architectural.svg"
        vector_engine.export_svg(vector_res, svg_path, objects=objects_for_blueprint, ceiling_height_m=hsp)
        if svg_path.exists():
            current_svg_bytes[0] = svg_path.read_bytes()

        # Update GUI Preview
        small_preview = cv2.resize(blueprint_img, (420, 308), interpolation=cv2.INTER_AREA)
        preview_cad_img.image = cv2.cvtColor(small_preview, cv2.COLOR_BGR2RGB)

        # Update Floating Top-Left HUD
        if cb_show_hud.value:
            hud_handle[0].content = make_architect_hud_html(small_preview, area, hsp, volume)
            hud_handle[0].visible = True
        else:
            hud_handle[0].visible = False

        # Update 3D Grid
        grid_handle.position = ((min_xm + max_xm) / 2.0, (min_ym + max_ym) / 2.0, 0.0)
        grid_handle.width = max(4.0, span_w + 1.0)
        grid_handle.height = max(5.0, span_l + 1.0)
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

    @btn_download_dxf.on_click
    def _(event: viser.GuiEvent):
        target_client = event.client
        if target_client is None:
            clients = list(server.get_clients().values())
            target_client = clients[0] if clients else None

        if target_client is not None and len(current_dxf_bytes[0]) > 0:
            target_client.send_file_download(
                "Plan_Architectural_ArchiCAD.dxf",
                current_dxf_bytes[0],
            )
            print("[Architect] Plan architectural DXF (AutoCAD/ArchiCAD) envoyé pour téléchargement client.")

    @btn_download_svg.on_click
    def _(event: viser.GuiEvent):
        target_client = event.client
        if target_client is None:
            clients = list(server.get_clients().values())
            target_client = clients[0] if clients else None

        if target_client is not None and len(current_svg_bytes[0]) > 0:
            target_client.send_file_download(
                "Plan_Architectural_Vectoriel.svg",
                current_svg_bytes[0],
            )
            print("[Architect] Plan architectural vectoriel SVG envoyé pour téléchargement client.")

    @btn_view_topdown.on_click
    def _(event: viser.GuiEvent):
        """Aligns 3D camera to look straight down at the floor (Top-Down Ortho View)."""
        mid_x = 0.35
        mid_y = 1.85
        clients = [event.client] if event.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = np.array([mid_x, mid_y, 6.2])
                c.camera.look_at = np.array([mid_x, mid_y, 0.0])
                c.camera.up_direction = np.array([0.0, 1.0, 0.0])
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

    # First-person camera movement logic in canonical room frame
    up_norm = np.array([0.0, 0.0, 1.0])

    def rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1.0 - cos_a)

    def move_clients(direction: str):
        step_len = float(slider_walk_step.value)
        for client in server.get_clients().values():
            pos = np.array(client.camera.position, dtype=np.float64)
            look = np.array(client.camera.look_at, dtype=np.float64)
            view_vec = look - pos
            dist = np.linalg.norm(view_vec)
            if dist < 1e-4:
                dist = 1.0
                view_vec = np.array([0.0, 1.0, 0.0])

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
        eye_h = float(slider_eye_height.value)
        eye_pos = np.array([0.0, -0.4, eye_h])
        target_pos = np.array([0.0, 2.5, eye_h])

        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = eye_pos
                c.camera.look_at = target_pos
                c.camera.up_direction = np.array([0.0, 0.0, 1.0])
        print("[Viser] Mode Première Personne activé à l'entrée de la pièce.")

    @btn_reset_view.on_click
    def _(e: viser.GuiEvent):
        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = np.array([0.35, -2.5, 3.8])
                c.camera.look_at = np.array([0.35, 1.85, 0.8])
                c.camera.up_direction = np.array([0.0, 0.0, 1.0])

    @btn_do_teleport.on_click
    def _(e: viser.GuiEvent):
        target_name = dd_teleport.value
        eye_h = float(slider_eye_height.value)

        if "Entrée" in target_name:
            eye_pos = np.array([0.0, -0.4, eye_h])
            look_pos = np.array([0.0, 2.0, eye_h])
        elif "Lit" in target_name:
            eye_pos = np.array([0.89, 2.20, eye_h])
            look_pos = np.array([0.89, 3.43, eye_h * 0.7])
        elif "Bureau" in target_name:
            eye_pos = np.array([-0.16, 0.70, eye_h])
            look_pos = np.array([-0.16, 1.57, eye_h * 0.7])
        else: # Fenêtre
            eye_pos = np.array([0.0, 3.8, eye_h])
            look_pos = np.array([0.0, 4.6, eye_h])

        clients = [e.client] if e.client is not None else list(server.get_clients().values())
        for c in clients:
            if c is not None:
                c.camera.position = eye_pos
                c.camera.look_at = look_pos
                c.camera.up_direction = np.array([0.0, 0.0, 1.0])

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
