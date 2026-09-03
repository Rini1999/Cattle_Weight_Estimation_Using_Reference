import streamlit as st
import numpy as np
import pandas as pd
import joblib

from PIL import Image, ImageOps

from segmentation import load_segmentation_models
from keypoint_detection import load_keypoint_models
from feature_extraction import extract_features

import importlib.metadata as md

st.write("fastai", md.version("fastai"))
st.write("fastcore", md.version("fastcore"))
st.write("fasttransform", md.version("fasttransform"))
st.stop()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Cattle Weight Estimator",
    layout="wide"
)

# ---------------------------------------------------
# CACHE MODELS
# ---------------------------------------------------

@st.cache_resource
def load_all_models():

    print("STEP 1: Starting load_all_models()", flush=True)

    print("STEP 2: Loading segmentation models...", flush=True)
    side_seg_model, rear_seg_model = load_segmentation_models()

    print("STEP 3: Segmentation models loaded.", flush=True)

    print("STEP 4: Loading keypoint models...", flush=True)
    side_kp_model, rear_kp_model = load_keypoint_models()

    print("STEP 5: Keypoint models loaded.", flush=True)

    print("STEP 6: Loading regression model...", flush=True)
    reg_model = joblib.load(
        "models/Overall_best_model_with_reference_catboost.joblib"
    )

    print("STEP 7: Regression model loaded.", flush=True)

    return (
        side_seg_model,
        rear_seg_model,
        side_kp_model,
        rear_kp_model,
        reg_model
    )


(
    side_seg_model,
    rear_seg_model,
    side_kp_model,
    rear_kp_model,
    model
) = load_all_models()

# ---------------------------------------------------
# IMAGE STANDARDIZATION
# ---------------------------------------------------

def prepare_uploaded_image(uploaded_file):
    """
    Preserve original resolution.
    Fix EXIF orientation.
    Convert to RGB.
    """
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img


def check_image_quality(img, view_name):
    """
    Warn users if the uploaded image appears compressed or resized.
    Does NOT resize the image.
    """
    width, height = img.size
    megapixels = width * height / 1_000_000

    warnings = []

    if megapixels < 1.0:
        warnings.append(
            f"⚠️ {view_name} image resolution is very low ({width}×{height}). "
            "Predictions may be inaccurate."
        )

    if width < 800 or height < 600:
        warnings.append(
            f"⚠️ {view_name} image appears resized or compressed."
        )

    aspect_ratio = width / height

    if aspect_ratio < 0.4 or aspect_ratio > 3.5:
        warnings.append(
            f"⚠️ {view_name} image has an unusual aspect ratio ({aspect_ratio:.2f})."
        )

    return warnings


# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("Cattle Weight Estimator")

st.write("""
Upload:

- **One SIDE-view image**
- **One REAR-view image**

**Important:** Upload the original camera images whenever possible. Avoid screenshots,
social media images, or compressed copies because they can reduce prediction accuracy.
""")

col1, col2 = st.columns(2)

side_file = col1.file_uploader(
    "Upload Side View",
    type=["jpg", "jpeg", "png"]
)

rear_file = col2.file_uploader(
    "Upload Rear View",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------

if side_file and rear_file:

    side_img = prepare_uploaded_image(side_file)
    rear_img = prepare_uploaded_image(rear_file)

    col1.image(
        side_img,
        caption=f"Side View ({side_img.size[0]} × {side_img.size[1]})",
        width="stretch"
    )

    col2.image(
        rear_img,
        caption=f"Rear View ({rear_img.size[0]} × {rear_img.size[1]})",
        width="stretch"
    )

    # Show quality warnings
    side_warnings = check_image_quality(side_img, "Side")
    rear_warnings = check_image_quality(rear_img, "Rear")

    if side_warnings or rear_warnings:
        st.warning(
            "The uploaded images may have been resized or compressed. "
            "For best accuracy, upload the original camera images."
        )

        for msg in side_warnings + rear_warnings:
            st.write(msg)

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    if st.button("Predict Weight"):

        try:

            # --------------------------------------------
            # SIDE FEATURES
            # --------------------------------------------

            with st.spinner("Extracting side-view features..."):

                print("STEP 8: Extracting SIDE features...", flush=True)

                side_feats = extract_features(
                    side_img,
                    "Side",
                    side_seg_model,
                    side_kp_model
                )

                print("STEP 9: SIDE extraction completed.", flush=True)

            if side_feats is None:
                st.error("Side-view feature extraction failed.")
                st.stop()

            # --------------------------------------------
            # REAR FEATURES
            # --------------------------------------------

            with st.spinner("Extracting rear-view features..."):

                print("STEP 10: Extracting REAR features...", flush=True)

                rear_feats = extract_features(
                    rear_img,
                    "Rear",
                    rear_seg_model,
                    rear_kp_model
                )

                print("STEP 11: REAR extraction completed.", flush=True)

            if rear_feats is None:
                st.error("Rear-view feature extraction failed.")
                st.stop()

            # --------------------------------------------
            # DERIVED FEATURES
            # --------------------------------------------

            volume_proxy_1 = (
                side_feats["side_mask_area"] *
                rear_feats["rear_mask_area"]
            )

            volume_proxy_2 = (
                side_feats["side_convex_hull_area"] *
                rear_feats["rear_convex_hull_area"]
            )

            girth_proxy_1 = (
                rear_feats["rear_bbox_width"] *
                rear_feats["rear_bbox_height"]
            )

            girth_proxy_2 = (
                rear_feats["rear_major_axis_length"] *
                rear_feats["rear_minor_axis_length"]
            )

            compactness_side = (
                side_feats["side_mask_area"] /
                (side_feats["side_perimeter"] ** 2 + 1e-6)
            )

            compactness_rear = (
                rear_feats["rear_mask_area"] /
                (rear_feats["rear_perimeter"] ** 2 + 1e-6)
            )

            area_balance_ratio = (
                side_feats["side_mask_area"] /
                (rear_feats["rear_mask_area"] + 1e-6)
            )

            shape_balance_ratio = (
                side_feats["side_major_axis_length"] /
                (rear_feats["rear_major_axis_length"] + 1e-6)
            )

            aspect_balance_ratio = (
                side_feats["side_aspect_ratio"] /
                (rear_feats["rear_aspect_ratio"] + 1e-6)
            )

            eccentricity_balance = (
                side_feats["side_eccentricity"] /
                (rear_feats["rear_eccentricity"] + 1e-6)
            )

            # --------------------------------------------
            # FEATURE VECTOR
            # --------------------------------------------

            feature_dict = {

                "body_length_in": side_feats["body_length"],
                "chest_depth_in": side_feats["chest_depth"],
                "rear_depth_in": side_feats["rear_depth"],
                "hip_height_in": side_feats["hip_height"],
                "body_oblique_length_in": side_feats["body_oblique_length"],

                "rear_height_width_ratio":
                    rear_feats["rear_height_width_ratio"],

                "side_major_axis_length":
                    side_feats["side_major_axis_length"],

                "side_bbox_width":
                    side_feats["side_bbox_width"],

                "side_aspect_ratio":
                    side_feats["side_aspect_ratio"],

                "side_eccentricity":
                    side_feats["side_eccentricity"],

                "side_mask_area":
                    side_feats["side_mask_area"],

                "side_convex_hull_area":
                    side_feats["side_convex_hull_area"],

                "side_minor_axis_length":
                    side_feats["side_minor_axis_length"],

                "side_perimeter":
                    side_feats["side_perimeter"],

                "side_bbox_height":
                    side_feats["side_bbox_height"],

                "side_solidity":
                    side_feats["side_solidity"],

                "side_circularity":
                    side_feats["side_circularity"],

                "side_extent":
                    side_feats["side_extent"],

                "rear_major_axis_length":
                    rear_feats["rear_major_axis_length"],

                "rear_bbox_width":
                    rear_feats["rear_bbox_width"],

                "rear_aspect_ratio":
                    rear_feats["rear_aspect_ratio"],

                "rear_eccentricity":
                    rear_feats["rear_eccentricity"],

                "rear_mask_area":
                    rear_feats["rear_mask_area"],

                "rear_convex_hull_area":
                    rear_feats["rear_convex_hull_area"],

                "rear_minor_axis_length":
                    rear_feats["rear_minor_axis_length"],

                "rear_perimeter":
                    rear_feats["rear_perimeter"],

                "rear_bbox_height":
                    rear_feats["rear_bbox_height"],

                "rear_solidity":
                    rear_feats["rear_solidity"],

                "rear_circularity":
                    rear_feats["rear_circularity"],

                "rear_extent":
                    rear_feats["rear_extent"],

                "volume_proxy_1": volume_proxy_1,
                "volume_proxy_2": volume_proxy_2,
                "girth_proxy_1": girth_proxy_1,
                "girth_proxy_2": girth_proxy_2,
                "compactness_side": compactness_side,
                "compactness_rear": compactness_rear,
                "area_balance_ratio": area_balance_ratio,
                "shape_balance_ratio": shape_balance_ratio,
                "aspect_balance_ratio": aspect_balance_ratio,
                "eccentricity_balance": eccentricity_balance,
            }

            X = pd.DataFrame([feature_dict])

            print("STEP 12: Running CatBoost prediction...", flush=True)

            prediction = model.predict(X)[0]

            print("STEP 13: Prediction completed.", flush=True)

            st.success(f"### Estimated Weight: {prediction:.2f} kg")

        except Exception as e:

            print(f"ERROR: {e}", flush=True)

            st.error("An error occurred during prediction.")
            st.exception(e)