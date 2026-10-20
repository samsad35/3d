"""
Grounded-SAM Open-Vocabulary Segmentation & 3D Object Lifting Module
Detects open-vocabulary objects using Grounding DINO, segments with SAM,
and lifts 2D instances to 3D point cloud clusters with centroids and bounding boxes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


@dataclass
class DetectedObject3D:
    label: str
    confidence: float
    centroid_3d: np.ndarray        # (3,) in world frame
    bbox_3d_min: np.ndarray        # (3,)
    bbox_3d_max: np.ndarray        # (3,)
    points_3d: np.ndarray          # (K, 3) 3D points
    source_view: int               # Index of source image view
    mask_2d: np.ndarray            # (H, W) boolean mask


class GroundedSAMSegmenter:
    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Grounded-SAM] Loading Grounding DINO ({dino_model_id})...")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
        self.dino_model.eval()

        print(f"[Grounded-SAM] Loading SAM ({sam_model_id})...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()
        print("[Grounded-SAM] Models initialized successfully.")

    @torch.no_grad()
    def detect_and_segment_image(
        self,
        pil_image: Image.Image,
        text_queries: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Dict]:
        """
        Runs open-vocabulary detection + SAM segmentation on a single PIL Image.
        Returns list of dicts with: 'label', 'score', 'box' (xyxy in pixels), 'mask' (bool HxW).
        """
        w, h = pil_image.size
        # Grounding DINO expects period-separated queries: "chair . bed . table ."
        prompt = " . ".join(text_queries) + " ."

        inputs = self.dino_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)

        # Post-process detection results
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        if len(boxes) == 0:
            return []

        # SAM segmentation with box prompts
        # Format boxes for SAM: [[ [x1, y1, x2, y2], ... ]]
        sam_boxes = [[box.tolist() for box in boxes]]
        sam_inputs = self.sam_processor(pil_image, input_boxes=sam_boxes, return_tensors="pt").to(self.device)
        sam_outputs = self.sam_model(**sam_inputs)

        # Post-process SAM masks
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0] # Shape: (num_boxes, 3, H, W)

        # Take highest scoring mask per box (index 0)
        detections = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            mask_bool = masks[i, 0].numpy().astype(bool)
            detections.append({
                "label": label.strip(),
                "score": float(score),
                "box": box,
                "mask": mask_bool,
            })

        return detections

    def lift_detections_to_3d(
        self,
        image_paths: List[str],
        world_points_map: np.ndarray,      # (S, H, W, 3) from VGGT
        confidence_map: np.ndarray,        # (S, H, W)
        text_queries: List[str],
        min_cluster_points: int = 40,
        merge_distance_threshold: float = 0.40,
    ) -> List[DetectedObject3D]:
        """
        Detects objects across all views and lifts them into 3D world coordinates.
        Merges redundant detections of the same object across different views.
        """
        all_objects_3d: List[DetectedObject3D] = []
        s, vggt_h, vggt_w, _ = world_points_map.shape

        for view_idx, img_path in enumerate(image_paths):
            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            detections_2d = self.detect_and_segment_image(pil_img, text_queries)

            for det in detections_2d:
                mask_2d = det["mask"] # (orig_h, orig_w)
                
                # Resize mask to VGGT resolution (vggt_w, vggt_h)
                mask_resized = np.array(
                    Image.fromarray(mask_2d).resize((vggt_w, vggt_h), Image.NEAREST)
                )

                # Extract corresponding 3D points
                pts_view = world_points_map[view_idx] # (vggt_h, vggt_w, 3)
                conf_view = confidence_map[view_idx]  # (vggt_h, vggt_w)

                valid_pts_mask = mask_resized & (conf_view > np.percentile(conf_view, 25))
                obj_pts = pts_view[valid_pts_mask]

                if len(obj_pts) < min_cluster_points:
                    continue

                # Filter spatial outliers (IQR or median distance)
                centroid = np.median(obj_pts, axis=0)
                dists = np.linalg.norm(obj_pts - centroid, axis=1)
                inlier_pts = obj_pts[dists < np.percentile(dists, 85)]

                if len(inlier_pts) < min_cluster_points:
                    continue

                clean_centroid = np.mean(inlier_pts, axis=0)
                bbox_min = np.min(inlier_pts, axis=0)
                bbox_max = np.max(inlier_pts, axis=0)

                obj_3d = DetectedObject3D(
                    label=det["label"],
                    confidence=det["score"],
                    centroid_3d=clean_centroid,
                    bbox_3d_min=bbox_min,
                    bbox_3d_max=bbox_max,
                    points_3d=inlier_pts,
                    source_view=view_idx,
                    mask_2d=mask_2d,
                )
                all_objects_3d.append(obj_3d)

        # Merge redundant detections across views
        merged_objects = self._merge_multiview_objects(all_objects_3d, merge_distance_threshold)
        print(f"[Grounded-SAM] Extracted {len(merged_objects)} unique 3D objects in the scene.")
        return merged_objects

    @staticmethod
    def _merge_multiview_objects(
        objects: List[DetectedObject3D], distance_threshold: float
    ) -> List[DetectedObject3D]:
        """
        Clusters 3D detections with the same label if their centroids are close.
        """
        if not objects:
            return []

        merged: List[DetectedObject3D] = []
        used = [False] * len(objects)

        for i in range(len(objects)):
            if used[i]:
                continue

            cluster = [objects[i]]
            used[i] = True

            for j in range(i + 1, len(objects)):
                if not used[j] and objects[i].label.lower() == objects[j].label.lower():
                    dist = np.linalg.norm(objects[i].centroid_3d - objects[j].centroid_3d)
                    if dist < distance_threshold:
                        cluster.append(objects[j])
                        used[j] = True

            # Merge cluster
            all_pts = np.vstack([obj.points_3d for obj in cluster])
            best_obj = max(cluster, key=lambda x: x.confidence)
            merged.append(
                DetectedObject3D(
                    label=best_obj.label,
                    confidence=best_obj.confidence,
                    centroid_3d=np.mean(all_pts, axis=0),
                    bbox_3d_min=np.min(all_pts, axis=0),
                    bbox_3d_max=np.max(all_pts, axis=0),
                    points_3d=all_pts,
                    source_view=best_obj.source_view,
                    mask_2d=best_obj.mask_2d,
                )
            )

        return merged
