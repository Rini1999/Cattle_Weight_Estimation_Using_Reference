import numpy as np
import torch
import cv2
import pathlib
import os

from PIL import ImageOps
from fastai.vision.all import load_learner


# ---------------------------------------------------
# WINDOWS -> LINUX PATH COMPATIBILITY
# ---------------------------------------------------
# The FastAI .pkl models were saved on Windows and may
# contain pathlib.WindowsPath objects. Streamlit Cloud
# runs Linux, where WindowsPath cannot be instantiated.
#
# Convert WindowsPath references to PosixPath during
# model deserialization.

if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath


# ---------------------------------------------------
# CPU / PERFORMANCE SETTINGS
# ---------------------------------------------------

cv2.setNumThreads(0)

torch.set_num_threads(1)


# ---------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------

SIDE_MODEL_PATH = os.path.join(
    "models",
    "stage-1.pkl"
)

REAR_MODEL_PATH = os.path.join(
    "models",
    "stage-2.pkl"
)


# ---------------------------------------------------
# LOAD SEGMENTATION MODELS
# ---------------------------------------------------

def load_segmentation_models():

    print("Loading side segmentation model...", flush=True)

    if not os.path.exists(SIDE_MODEL_PATH):
        raise FileNotFoundError(
            f"Side segmentation model not found: "
            f"{SIDE_MODEL_PATH}"
        )

    side_seg_model = load_learner(
        SIDE_MODEL_PATH,
        cpu=True
    )

    print(
        "Side segmentation model loaded.",
        flush=True
    )

    print("Loading rear segmentation model...", flush=True)

    if not os.path.exists(REAR_MODEL_PATH):
        raise FileNotFoundError(
            f"Rear segmentation model not found: "
            f"{REAR_MODEL_PATH}"
        )

    rear_seg_model = load_learner(
        REAR_MODEL_PATH,
        cpu=True
    )

    print(
        "Rear segmentation model loaded.",
        flush=True
    )

    print(
        "Segmentation models loaded successfully.",
        flush=True
    )

    return side_seg_model, rear_seg_model


# ---------------------------------------------------
# CLEAN MASK
# ---------------------------------------------------

def keep_largest_component(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return mask

    largest = max(
        contours,
        key=cv2.contourArea
    )

    clean = np.zeros_like(mask)

    cv2.drawContours(
        clean,
        [largest],
        -1,
        1,
        thickness=-1
    )

    return clean


# ---------------------------------------------------
# MAIN SEGMENTATION FUNCTION
# ---------------------------------------------------

def get_segmentation_masks(
    model,
    image,
    view_type
):

    try:

        image = ImageOps.exif_transpose(image)

        print(
            f"Running segmentation for {view_type}...",
            flush=True
        )

        original_w, original_h = image.size

        print(
            f"Original image size = "
            f"{original_w} x {original_h}",
            flush=True
        )

        # ---------------------------------------------------
        # NO MANUAL RESIZING
        # ---------------------------------------------------

        inference_image = image

        # ---------------------------------------------------
        # FASTAI TEST DATALOADER
        # ---------------------------------------------------

        dl = model.dls.test_dl(
            [inference_image],
            bs=1
        )

        print(
            "Running inference...",
            flush=True
        )

        preds, _ = model.get_preds(
            dl=dl
        )

        print(
            "Inference completed.",
            flush=True
        )

        # ---------------------------------------------------
        # PREDICTION MASK
        # ---------------------------------------------------

        pred_mask = (
            preds[0]
            .argmax(dim=0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        print(
            f"Prediction mask shape = "
            f"{pred_mask.shape}",
            flush=True
        )

        print(
            f"Unique predicted classes = "
            f"{np.unique(pred_mask)}",
            flush=True
        )

        # ---------------------------------------------------
        # RESIZE BACK TO ORIGINAL SIZE
        # ---------------------------------------------------

        if (
            pred_mask.shape[1] != original_w
            or pred_mask.shape[0] != original_h
        ):

            pred_mask = cv2.resize(
                pred_mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            print(
                "Mask resized back to original size.",
                flush=True
            )

        # ---------------------------------------------------
        # GET VOCAB SAFELY
        # ---------------------------------------------------

        vocab = getattr(
            model.dls,
            "vocab",
            None
        )

        if vocab is None:

            vocab = getattr(
                model.dls.train_ds,
                "vocab",
                None
            )

        print(
            f"Model vocab = {vocab}",
            flush=True
        )

        # ---------------------------------------------------
        # CLASS INDEX RESOLUTION
        # ---------------------------------------------------

        sticker_idx = None

        if vocab is not None:

            # NumPy array case

            if isinstance(vocab, np.ndarray):

                vocab = vocab.tolist()

            # List case

            if isinstance(vocab, list):

                if "Cattle" in vocab:

                    cattle_idx = vocab.index(
                        "Cattle"
                    )

                else:

                    raise ValueError(
                        f"'Cattle' class not found "
                        f"in model vocabulary: {vocab}"
                    )

                if "Sticker" in vocab:

                    sticker_idx = vocab.index(
                        "Sticker"
                    )

            # FastAI CategoryMap case

            elif hasattr(vocab, "o2i"):

                if "Cattle" in vocab.o2i:

                    cattle_idx = vocab.o2i[
                        "Cattle"
                    ]

                else:

                    raise ValueError(
                        f"'Cattle' class not found "
                        f"in model vocabulary: {vocab.o2i}"
                    )

                if "Sticker" in vocab.o2i:

                    sticker_idx = vocab.o2i[
                        "Sticker"
                    ]

            else:

                # Fallback

                if view_type == "Side":

                    sticker_idx = 0
                    cattle_idx = 1

                else:

                    cattle_idx = 0

        else:

            # ---------------------------------------------------
            # MANUAL FALLBACKS
            # ---------------------------------------------------

            if view_type == "Side":

                # Sticker, Cattle,
                # Background, Void

                sticker_idx = 0
                cattle_idx = 1

            else:

                # Cattle,
                # Background,
                # Void

                cattle_idx = 0

        print(
            f"Cattle class index = "
            f"{cattle_idx}",
            flush=True
        )

        print(
            f"Sticker class index = "
            f"{sticker_idx}",
            flush=True
        )

        # ---------------------------------------------------
        # CATTLE MASK
        # ---------------------------------------------------

        cattle_mask = (
            pred_mask == cattle_idx
        ).astype(np.uint8)

        # ---------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------

        cattle_mask = cv2.morphologyEx(
            cattle_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        cattle_mask = keep_largest_component(
            cattle_mask
        )

        print(
            f"Cattle mask area = "
            f"{np.sum(cattle_mask)}",
            flush=True
        )

        # ---------------------------------------------------
        # STICKER MASK
        # ---------------------------------------------------

        sticker_mask = None

        if (
            view_type == "Side"
            and sticker_idx is not None
        ):

            sticker_mask = (
                pred_mask == sticker_idx
            ).astype(np.uint8)

            sticker_mask = keep_largest_component(
                sticker_mask
            )

            print(
                f"Sticker mask area = "
                f"{np.sum(sticker_mask)}",
                flush=True
            )

        print(
            f"Segmentation finished for "
            f"{view_type}",
            flush=True
        )

        return sticker_mask, cattle_mask

    except Exception as e:

        print(
            f"Segmentation Error "
            f"({view_type}): {e}",
            flush=True
        )

        raise


# ---------------------------------------------------
# SCALE ESTIMATION
# ---------------------------------------------------

def compute_scale_from_sticker(
    sticker_mask,
    sticker_size_in=4.0
):

    if sticker_mask is None:

        print(
            "Sticker mask is None",
            flush=True
        )

        return None

    sticker_mask = sticker_mask.astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        sticker_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        print(
            "No sticker contour found",
            flush=True
        )

        return None

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(cnt)

    sticker_px = max(w, h)

    print(
        f"Sticker pixels = {sticker_px}",
        flush=True
    )

    if sticker_px == 0:

        return None

    scale = (
        sticker_size_in /
        float(sticker_px)
    )

    print(
        f"Computed scale = {scale}",
        flush=True
    )

    return scale


# ---------------------------------------------------
# DISTANCE UTILITIES
# ---------------------------------------------------

def euclidean(
    p1,
    p2
):

    return np.linalg.norm(
        np.array(p1) -
        np.array(p2)
    )


def dist_in_inches(
    p1,
    p2,
    scale
):

    return euclidean(
        p1,
        p2
    ) * scale
