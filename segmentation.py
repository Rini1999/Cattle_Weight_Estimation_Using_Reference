import os
import pathlib

# ---------------------------------------------------
# CROSS-PLATFORM PATH COMPATIBILITY
# ---------------------------------------------------
#
# The fastai model files contain serialized pathlib
# objects from a different operating system.
#
# Windows  -> convert serialized PosixPath to WindowsPath
# Linux    -> convert serialized WindowsPath to PosixPath
#
# Do NOT replace pathlib.Path itself.
# ---------------------------------------------------

if os.name == "nt":
    # Running on Windows
    pathlib.PosixPath = pathlib.WindowsPath
else:
    # Running on Linux / Streamlit Community Cloud
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import torch
import cv2

from PIL import ImageOps
from fastai.vision.all import load_learner


# ---------------------------------------------------
# CPU / PERFORMANCE SETTINGS
# ---------------------------------------------------

cv2.setNumThreads(0)

torch.set_num_threads(1)


# ---------------------------------------------------
# CROSS-PLATFORM FASTAI MODEL LOADER
# ---------------------------------------------------

def load_learner_cross_platform(model_path):

    print(f"Loading model: {model_path}", flush=True)

    try:

        model = load_learner(
            model_path,
            cpu=True
        )

        print(
            f"Successfully loaded: {model_path}",
            flush=True
        )

        return model

    except Exception as e:

        print(
            f"Failed to load model: {model_path}",
            flush=True
        )

        print(
            f"Error type: {type(e).__name__}",
            flush=True
        )

        print(
            f"Error message: {e}",
            flush=True
        )

        raise


# ---------------------------------------------------
# LOAD SEGMENTATION MODELS
# ---------------------------------------------------

def load_segmentation_models():

    print(
        "Loading side segmentation model...",
        flush=True
    )

    side_seg_model = load_learner_cross_platform(
        "models/stage-1.pkl"
    )

    print(
        "Loading rear segmentation model...",
        flush=True
    )

    rear_seg_model = load_learner_cross_platform(
        "models/stage-2.pkl"
    )

    print(
        "Segmentation models loaded.",
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
            or
            pred_mask.shape[0] != original_h
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
        cattle_idx = None

        if vocab is not None:

            # numpy array case

            if isinstance(vocab, np.ndarray):

                vocab = vocab.tolist()

            # list / tuple case

            if isinstance(vocab, (list, tuple)):

                if "Cattle" in vocab:

                    cattle_idx = vocab.index(
                        "Cattle"
                    )

                if "Sticker" in vocab:

                    sticker_idx = vocab.index(
                        "Sticker"
                    )

            # fastai CategoryMap case

            elif hasattr(vocab, "o2i"):

                if "Cattle" in vocab.o2i:

                    cattle_idx = vocab.o2i[
                        "Cattle"
                    ]

                if "Sticker" in vocab.o2i:

                    sticker_idx = vocab.o2i[
                        "Sticker"
                    ]

        # ---------------------------------------------------
        # MANUAL FALLBACKS
        # ---------------------------------------------------

        if cattle_idx is None:

            if view_type == "Side":

                # Sticker, Cattle,
                # Background, Void

                cattle_idx = 1

                if sticker_idx is None:

                    sticker_idx = 0

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
            np.ones(
                (3, 3),
                np.uint8
            )
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
            and
            sticker_idx is not None
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

        return None, None


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

