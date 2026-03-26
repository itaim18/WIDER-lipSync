"""
Occlusion fix: composite non-face objects (plate, cookies, hands) from the
original video back on top of the MuseTalk lip-synced output.

Approach:
1. Compute per-pixel difference between original and MuseTalk frames
2. Where difference is significant = face replacement zone
3. Within that zone, use BiSeNet face parsing on the original to identify face vs non-face
4. Non-face pixels in the replacement zone = occluding objects (plate, cookies)
5. Restore those from the original with soft edge blending
"""

import os
import sys
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Monkey-patch torch.load for PyTorch 2.7+
_orig_load = torch.load
def _patched_load(*a, **kw):
    if 'weights_only' not in kw:
        kw['weights_only'] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

# Add MuseTalk utils to path for BiSeNet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MuseTalk'))
from musetalk.utils.face_parsing.model import BiSeNet


class FaceParser:
    """Lightweight BiSeNet face parser for occlusion detection."""
    def __init__(self, resnet_path, model_path, device='cuda'):
        self.device = device
        self.net = BiSeNet(resnet_path)
        if device == 'cuda':
            self.net.cuda()
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def parse(self, image_bgr, size=(512, 512)):
        """
        Returns raw parsing labels (0-18) for the input image.
        Labels: 0=background, 1=skin, 2-3=brows, 4-5=eyes, 6=glasses,
        7-8=ears, 9=earring, 10=nose, 11=mouth_inner, 12=upper_lip,
        13=lower_lip, 14=neck, 15=necklace, 16=cloth, 17=hair, 18=hat
        """
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(size, Image.BILINEAR)
        img_tensor = self.preprocess(img_pil).unsqueeze(0)
        if self.device == 'cuda':
            img_tensor = img_tensor.cuda()
        out = self.net(img_tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # Resize back to original
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return parsing


def compute_occlusion_mask(original, musetalk, parser, diff_threshold=12, face_labels=None):
    """
    Compute a mask of non-face pixels that were modified by MuseTalk.
    These are occluding objects that should be restored from the original.
    """
    if face_labels is None:
        # All face-related labels (skin, brows, eyes, nose, lips, etc.)
        face_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

    # 1. Find pixels that MuseTalk changed
    diff = np.abs(original.astype(np.float32) - musetalk.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    modified_mask = (diff_gray > diff_threshold).astype(np.uint8)

    # 2. Clean up the modified mask (remove noise, fill gaps)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    modified_mask = cv2.morphologyEx(modified_mask, cv2.MORPH_CLOSE, kernel_close)
    modified_mask = cv2.morphologyEx(modified_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 3. Run face parsing on the original frame
    parsing = parser.parse(original)

    # 4. Create face mask (pixels that ARE face in the original)
    face_mask = np.isin(parsing, list(face_labels)).astype(np.uint8)

    # 5. Occluding pixels = modified by MuseTalk AND NOT face in original
    # These are plate, cookies, hands, clothes that got overwritten
    occlusion_mask = (modified_mask & (~face_mask.astype(bool)).astype(np.uint8))

    # Also exclude hair (17) and hat (18) from occlusion - those aren't objects in front
    # But include background (0), cloth (16), necklace (15) as potential occluders
    # Actually, we want to be conservative: only restore pixels that are clearly
    # non-face AND were modified. The parsing already handles this.

    # 6. Dilate the occlusion mask slightly to capture edge pixels
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occlusion_mask = cv2.dilate(occlusion_mask, kernel_dilate, iterations=1)

    # 7. Soft edges via Gaussian blur for smooth blending
    occlusion_soft = cv2.GaussianBlur(occlusion_mask.astype(np.float32) * 255,
                                       (15, 15), 4.0) / 255.0

    return occlusion_soft


def process_video(original_path, musetalk_path, output_path, resnet_path, model_path,
                  diff_threshold=12, temp_dir='temp_occlusion'):
    """Process the full video with occlusion fix."""

    os.makedirs(temp_dir, exist_ok=True)

    # Extract frames from both videos
    orig_dir = os.path.join(temp_dir, 'orig')
    muse_dir = os.path.join(temp_dir, 'muse')
    out_dir = os.path.join(temp_dir, 'out')
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(muse_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Get video properties
    cap = cv2.VideoCapture(musetalk_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Extract frames
    print("Extracting frames...")
    os.system(f"ffmpeg -y -v warning -i {original_path} -start_number 0 {orig_dir}/%08d.png")
    os.system(f"ffmpeg -y -v warning -i {musetalk_path} -start_number 0 {muse_dir}/%08d.png")

    # Initialize face parser
    print("Loading face parser...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = FaceParser(resnet_path, model_path, device=device)

    # Process each frame
    orig_frames = sorted(glob.glob(os.path.join(orig_dir, '*.png')))
    muse_frames = sorted(glob.glob(os.path.join(muse_dir, '*.png')))
    num_frames = min(len(orig_frames), len(muse_frames))

    print(f"Processing {num_frames} frames with occlusion fix...")
    for i in tqdm(range(num_frames)):
        orig = cv2.imread(orig_frames[i])
        muse = cv2.imread(muse_frames[i])

        # Ensure same size
        if orig.shape != muse.shape:
            muse = cv2.resize(muse, (orig.shape[1], orig.shape[0]))

        # Compute occlusion mask and blend
        occlusion = compute_occlusion_mask(orig, muse, parser, diff_threshold=diff_threshold)

        # Composite: restore occluding objects from original
        occlusion_3ch = occlusion[:, :, np.newaxis]
        result = (muse.astype(np.float32) * (1 - occlusion_3ch) +
                  orig.astype(np.float32) * occlusion_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f'{i:08d}.png'), result)

    # Re-encode video with audio from MuseTalk output
    print("Encoding output video...")
    temp_vid = os.path.join(temp_dir, 'temp_video.mp4')
    os.system(f"ffmpeg -y -v warning -r {fps} -f image2 -i {out_dir}/%08d.png "
              f"-vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid}")
    os.system(f"ffmpeg -y -v warning -i {temp_vid} -i {musetalk_path} "
              f"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}")

    print(f"Output saved to {output_path}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True, help='Path to original video')
    parser.add_argument('--musetalk', type=str, required=True, help='Path to MuseTalk output video')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--diff_threshold', type=int, default=12,
                        help='Pixel difference threshold for detecting modifications (lower=more sensitive)')
    parser.add_argument('--resnet_path', type=str,
                        default='./MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth')
    parser.add_argument('--model_path', type=str,
                        default='./MuseTalk/models/face-parse-bisent/79999_iter.pth')
    args = parser.parse_args()

    process_video(
        original_path=args.original,
        musetalk_path=args.musetalk,
        output_path=args.output,
        resnet_path=args.resnet_path,
        model_path=args.model_path,
        diff_threshold=args.diff_threshold,
    )
