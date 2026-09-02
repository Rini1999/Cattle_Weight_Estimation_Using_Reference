
import os
import sys
import pathlib

# ---------------------------------------------------
# ENVIRONMENT DEBUG
# ---------------------------------------------------
import fastai
import fastcore
import torch

print("========== ENVIRONMENT DEBUG ==========", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"fastai version: {fastai.__version__}", flush=True)
print(f"fastai location: {fastai.__file__}", flush=True)
print(f"fastcore location: {fastcore.__file__}", flush=True)
print(f"torch version: {torch.__version__}", flush=True)
print(f"torch location: {torch.__file__}", flush=True)
print("========================================", flush=True)

# ---------------------------------------------------
# CROSS-PLATFORM PATH FIX
# ---------------------------------------------------
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import cv2
from PIL import ImageOps

from fastai.vision.all import load_learner
from fastai.vision.core import PILImage

# ---------------------------------------------------
# CPU SETTINGS
# ---------------------------------------------------
DEVICE = torch.device("cpu")
cv2.setNumThreads(0)
torch.set_num_threads(1)

# ---------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------
def load_learner_cross_platform(model_path):

    print(f"Loading model: {model_path}", flush=True)

    learner = load_learner(model_path, cpu=True)

    learner.model.to(DEVICE)
    learner.model.eval()

    print(f"Successfully loaded: {model_path}", flush=True)

    return learner


def load_segmentation_models():

    print("Loading side segmentation model...", flush=True)
    side_model = load_learner_cross_platform("models/stage-1.pkl")

    print("Loading rear segmentation model...", flush=True)
    rear_model = load_learner_cross_platform("models/stage-2.pkl")

    print("Segmentation models loaded.", flush=True)

    return side_model, rear_model


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

    cv2.drawContours(clean, [largest], -1, 1, thickness=-1)

    return clean


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

        # ---------------------------------------------------
        # CREATE FASTAI IMAGE (NO DATALOADER)
        # ---------------------------------------------------
        print("Preparing FastAI image...", flush=True)

        img = PILImage.create(image)

        # Apply the saved validation transforms directly.
        x = model.dls.valid.after_item(img)

        # Convert TensorImage -> Tensor and add batch dimension.
        x = x.unsqueeze(0).to(DEVICE)

        print(f"Input tensor shape = {x.shape}", flush=True)

        # ---------------------------------------------------
        # FORWARD PASS
        # ---------------------------------------------------
        print("Running model forward pass...", flush=True)

        with torch.no_grad():
            output = model.model(x)

        print("Forward pass completed.", flush=True)
        print(f"Output tensor shape = {output.shape}", flush=True)

        # ---------------------------------------------------
        # PREDICTION MASK
        # ---------------------------------------------------
        pred_mask = (
            output[0]
            .argmax(dim=0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        print(
            f"Prediction mask shape = {pred_mask.shape}",
            flush=True
        )

        print(
            f"Unique classes = {np.unique(pred_mask)}",
            flush=True
        )

        # ---------------------------------------------------
        # RESIZE TO ORIGINAL IMAGE
        # ---------------------------------------------------
        if pred_mask.shape != (original_h, original_w):

            pred_mask = cv2.resize(
                pred_mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            print(
                "Mask resized to original image size.",
                flush=True
            )

        # ---------------------------------------------------
        # CLASS INDICES
        # ---------------------------------------------------
        if view_type == "Side":
            sticker_idx = 0
            cattle_idx = 1
        else:
            sticker_idx = None
            cattle_idx = 0

        # ---------------------------------------------------
        # CATTLE MASK
        # ---------------------------------------------------
        cattle_mask = (pred_mask == cattle_idx).astype(np.uint8)

        cattle_mask = cv2.morphologyEx(
            cattle_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        cattle_mask = keep_largest_component(cattle_mask)

        cattle_area = int(cattle_mask.sum())

        print(f"Cattle mask area = {cattle_area}", flush=True)

        if cattle_area == 0:
            print("ERROR: Empty cattle mask.", flush=True)
            return None, None

        # ---------------------------------------------------
        # STICKER MASK
        # ---------------------------------------------------
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

        print(
            f"Segmentation Error ({view_type}): {type(e).__name__}: {e}",
            flush=True
        )

        import traceback
        traceback.print_exc()

        return None, None


# ---------------------------------------------------
# SCALE ESTIMATION
# ---------------------------------------------------
def compute_scale_from_sticker(
    sticker_mask,
    sticker_size_in=4.0
):

    if sticker_mask is None:
        print("Sticker mask is None", flush=True)
        return None

    sticker_mask = sticker_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        sticker_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No sticker contour found", flush=True)
        return None

    cnt = max(contours, key=cv2.contourArea)

    _, _, w, h = cv2.boundingRect(cnt)

    sticker_px = max(w, h)

    print(f"Sticker pixels = {sticker_px}", flush=True)

    if sticker_px == 0:
        return None

    scale = sticker_size_in / float(sticker_px)

    print(f"Computed scale = {scale}", flush=True)

    return scale


# ---------------------------------------------------
# DISTANCE UTILITIES
# ---------------------------------------------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def dist_in_inches(p1, p2, scale):
    return euclidean(p1, p2) * scale