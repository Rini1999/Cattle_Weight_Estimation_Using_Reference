import os
import pathlib

# ---------------------------------------------------
# CROSS-PLATFORM PATH FIX
# ---------------------------------------------------
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import cv2
import torch

from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from fastai.vision.all import load_learner

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------
cv2.setNumThreads(0)
torch.set_num_threads(1)

device = torch.device("cpu")

TARGET_W = 480
TARGET_H = 360

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

def load_learner_cross_platform(model_path):
    print(f"Loading model: {model_path}", flush=True)

    learner = load_learner(model_path, cpu=True)
    learner.model.eval()

    print(f"Successfully loaded: {model_path}", flush=True)

    return learner


def load_segmentation_models():
    print("Loading side segmentation model...", flush=True)
    side = load_learner_cross_platform("models/stage-1.pkl")

    print("Loading rear segmentation model...", flush=True)
    rear = load_learner_cross_platform("models/stage-2.pkl")

    print("Segmentation models loaded.", flush=True)

    return side, rear

# ---------------------------------------------------
# KEEP LARGEST COMPONENT
# ---------------------------------------------------

def keep_largest_component(mask):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)

    clean = np.zeros_like(mask)

    cv2.drawContours(clean, [largest], -1, 1, -1)

    return clean

# ---------------------------------------------------
# LETTERBOX RESIZE
# ---------------------------------------------------

def resize_with_padding(image):
    """
    Resize image while preserving aspect ratio and pad to 480x360.
    Returns padded image and padding information.
    """

    original_w, original_h = image.size

    scale = min(
        TARGET_W / original_w,
        TARGET_H / original_h
    )

    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))

    pad_x = (TARGET_W - new_w) // 2
    pad_y = (TARGET_H - new_h) // 2

    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y, new_w, new_h

# ---------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------

def get_segmentation_masks(model, image, view_type):

    try:

        image = ImageOps.exif_transpose(image)

        print(f"Running segmentation for {view_type}...", flush=True)

        original_w, original_h = image.size

        print(
            f"Original image size = {original_w} x {original_h}",
            flush=True
        )

        # ---------------------------------------------
        # Resize with padding
        # ---------------------------------------------

        padded_img, scale, pad_x, pad_y, new_w, new_h = \
            resize_with_padding(image)

        print("Preparing segmentation input...", flush=True)

        x = ToTensor()(padded_img).unsqueeze(0).to(device)

        print(f"Input tensor shape = {x.shape}", flush=True)

        # ---------------------------------------------
        # Forward pass
        # ---------------------------------------------

        with torch.no_grad():
            output = model.model(x)

        pred_mask = (
            output[0]
            .argmax(dim=0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        print(f"Prediction mask shape = {pred_mask.shape}", flush=True)

        # ---------------------------------------------
        # Remove padding
        # ---------------------------------------------

        pred_mask = pred_mask[
            pad_y: pad_y + new_h,
            pad_x: pad_x + new_w
        ]

        # ---------------------------------------------
        # Resize back to original resolution
        # ---------------------------------------------

        pred_mask = cv2.resize(
            pred_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        print(
            f"Mask restored to original size = {pred_mask.shape}",
            flush=True
        )

        # ---------------------------------------------
        # Class indices
        # ---------------------------------------------

        if view_type == "Side":
            sticker_idx = 0
            cattle_idx = 1
        else:
            sticker_idx = None
            cattle_idx = 0

        cattle_mask = (pred_mask == cattle_idx).astype(np.uint8)
        cattle_mask = keep_largest_component(cattle_mask)

        print(
            f"Cattle mask area = {int(cattle_mask.sum())}",
            flush=True
        )

        if cattle_mask.sum() == 0:
            return None, None

        sticker_mask = None

        if view_type == "Side":
            sticker_mask = (pred_mask == sticker_idx).astype(np.uint8)
            sticker_mask = keep_largest_component(sticker_mask)

            print(
                f"Sticker mask area = {int(sticker_mask.sum())}",
                flush=True
            )

        print(f"Segmentation finished for {view_type}", flush=True)

        return sticker_mask, cattle_mask

    except Exception as e:

        import traceback

        print(
            f"Segmentation Error ({view_type}): {type(e).__name__}: {e}",
            flush=True
        )

        traceback.print_exc()

        return None, None

# ---------------------------------------------------
# SCALE
# ---------------------------------------------------

def compute_scale_from_sticker(sticker_mask, sticker_size_in=4.0):

    if sticker_mask is None:
        return None

    contours, _ = cv2.findContours(
        sticker_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    _, _, w, h = cv2.boundingRect(cnt)

    sticker_px = max(w, h)

    if sticker_px == 0:
        return None

    scale = sticker_size_in / float(sticker_px)

    print(f"Sticker pixels = {sticker_px}", flush=True)
    print(f"Computed scale = {scale}", flush=True)

    return scale

# ---------------------------------------------------
# DISTANCE UTILITIES
# ---------------------------------------------------

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def dist_in_inches(p1, p2, scale):
    return euclidean(p1, p2) * scale