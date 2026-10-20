import cv2
import os
from pathlib import Path

def extract_keyframes(video_path: str, output_dir: str, num_frames: int = 24, max_dim: int = 1024):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        raise ValueError(f"Could not open video {video_path}")
        
    print(f"Extracting {num_frames} frames from total {total_frames} frames...")
    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    saved_paths = []
    for count, frame_id in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        h, w = frame.shape[:2]
        # Resize maintaining aspect ratio
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        frame_name = f"frame_{count:03d}.jpg"
        save_file = str(out_path / frame_name)
        cv2.imwrite(save_file, frame)
        saved_paths.append(save_file)
        
    cap.release()
    print(f"Successfully saved {len(saved_paths)} frames to {output_dir}")
    return saved_paths

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent
    extract_keyframes(
        video_path=str(ROOT_DIR / "data" / "custom_video.mp4"),
        output_dir=str(ROOT_DIR / "data" / "custom_frames"),
        num_frames=80,
        max_dim=1024
    )
