import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch

INSIGHTFACE_ROOT = "/home/itaim/wider-itai/LatentSync/checkpoints/auxiliary"


class FaceCropper:
    """Face and mouth region detection and cropping using InsightFace."""

    def __init__(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        if not torch.cuda.is_available():
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=INSIGHTFACE_ROOT,
            providers=providers,
        )
        # 720x1280 videos with a single centered face do not need a large
        # detection window, and a smaller size makes the CPU-only path tractable.
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))

    def get_face(self, frame):
        """Detect the primary face in a frame. Returns face object or None."""
        faces = self.app.get(frame)
        if not faces:
            return None
        return faces[0]

    def extract_crops(self, frames, detect_every=5):
        """
        For each frame, detect face, crop face and mouth regions.
        Returns dict: {"face_crops": [...], "mouth_crops": [...]}
        """
        face_crops = []
        mouth_crops = []
        last_face = None

        for idx, frame in enumerate(frames):
            face = None
            if idx % detect_every == 0 or last_face is None:
                face = self.get_face(frame)
                if face is not None:
                    last_face = face
            else:
                face = last_face

            if face is None:
                continue

            h, w = frame.shape[:2]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Face crop with 10% padding
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            fx1 = max(0, x1 - pad_x)
            fy1 = max(0, y1 - pad_y)
            fx2 = min(w, x2 + pad_x)
            fy2 = min(h, y2 + pad_y)
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size > 0:
                face_crops.append(face_crop)

            # Mouth crop using 106-point landmarks if available
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106
                mouth_pts = lm[52:72]
                mx1 = int(mouth_pts[:, 0].min()) - 10
                my1 = int(mouth_pts[:, 1].min()) - 10
                mx2 = int(mouth_pts[:, 0].max()) + 10
                my2 = int(mouth_pts[:, 1].max()) + 10
            else:
                face_h = y2 - y1
                my1 = y1 + int(face_h * 0.55)
                my2 = y2 + 5
                mx1 = x1 - 5
                mx2 = x2 + 5

            mx1 = max(0, mx1)
            my1 = max(0, my1)
            mx2 = min(w, mx2)
            my2 = min(h, my2)
            mouth_crop = frame[my1:my2, mx1:mx2]
            if mouth_crop.size > 0:
                mouth_crops.append(mouth_crop)

        return {"face_crops": face_crops, "mouth_crops": mouth_crops}
