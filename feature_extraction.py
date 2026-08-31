import numpy as np
import cv2

from PIL import ImageOps

from segmentation import *
from keypoint_detection import *


# ---------------------------------------------------
# DISTANCE FUNCTIONS
# ---------------------------------------------------

def euclidean(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def dist_in_inches(p1, p2, scale):

    return euclidean(
        p1,
        p2
    ) * scale


# ---------------------------------------------------
# MAIN FEATURE EXTRACTION
# ---------------------------------------------------

def extract_features(
    image,
    view_type,
    seg_model,
    kp_model
):

    # ---------------------------------------------------
    # FIX IMAGE ORIENTATION
    # ---------------------------------------------------

    image = ImageOps.exif_transpose(image)

    # ---------------------------------------------------
    # SEGMENTATION
    # ---------------------------------------------------

    sticker_mask, cattle_mask = \
        get_segmentation_masks(
            seg_model,
            image,
            view_type
        )

    if cattle_mask is None:

        print("Cattle mask is None")

        return None

    cattle_mask = cattle_mask.astype(np.uint8)

    # ---------------------------------------------------
    # SCALE
    # ---------------------------------------------------

    if view_type == "Side":

        scale = compute_scale_from_sticker(
            sticker_mask
        )

        if scale is None:

            print("Scale computation failed.")

            return None

        print(f"Computed scale = {scale}")

    else:

        scale = 1.0

    # ---------------------------------------------------
    # FIND CONTOUR
    # ---------------------------------------------------

    contours, _ = cv2.findContours(
        cattle_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        print("No cattle contour found.")

        return None

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    # ---------------------------------------------------
    # BASIC SHAPE FEATURES
    # ---------------------------------------------------

    x, y, w, h = cv2.boundingRect(cnt)

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)

    solidity = (
        area / (hull_area + 1e-6)
    )

    extent = (
        area / (w * h + 1e-6)
    )

    circularity = (
        (4 * np.pi * area) /
        (perimeter ** 2 + 1e-6)
    )

    # ---------------------------------------------------
    # ELLIPSE FEATURES
    # ---------------------------------------------------

    if len(cnt) >= 5:

        (_, _), (MA, ma), _ = \
            cv2.fitEllipse(cnt)

        major_axis = max(MA, ma)
        minor_axis = min(MA, ma)

    else:

        major_axis = max(w, h)
        minor_axis = min(w, h)

    eccentricity = np.sqrt(
        max(
            0,
            1 - (
                (minor_axis ** 2) /
                (major_axis ** 2 + 1e-6)
            )
        )
    )

    # ---------------------------------------------------
    # ASPECT RATIO
    # ---------------------------------------------------

    if view_type == "Side":

        # TRAINING:
        # side_aspect_ratio = width / height

        aspect_ratio = (
            w / (h + 1e-6)
        )

    else:

        # TRAINING:
        # rear_aspect_ratio = height / width

        aspect_ratio = (
            h / (w + 1e-6)
        )

    # ---------------------------------------------------
    # KEYPOINTS
    # ---------------------------------------------------

    kp = get_keypoints(
        kp_model,
        image
    )

    # ---------------------------------------------------
    # SIDE FEATURES
    # ---------------------------------------------------

    if view_type == "Side":

        body_length = \
            dist_in_inches(
                kp[0],
                kp[1],
                scale
            )

        chest_depth = \
            dist_in_inches(
                kp[3],
                kp[4],
                scale
            )

        rear_depth = \
            dist_in_inches(
                kp[7],
                kp[8],
                scale
            )

        hip_height = \
            dist_in_inches(
                kp[5],
                kp[6],
                scale
            )

        body_oblique_length = \
            dist_in_inches(
                kp[2],
                kp[1],
                scale
            )

        print(f"Body length = {body_length}")
        print(f"Chest depth = {chest_depth}")
        print(f"Rear depth = {rear_depth}")
        print(f"Hip height = {hip_height}")

        return {

            # -----------------------------
            # LINEAR FEATURES
            # -----------------------------

            "body_length":
                body_length,

            "chest_depth":
                chest_depth,

            "rear_depth":
                rear_depth,

            "hip_height":
                hip_height,

            "body_oblique_length":
                body_oblique_length,

            # -----------------------------
            # SHAPE FEATURES
            # -----------------------------

            "side_mask_area":
                area,

            "side_convex_hull_area":
                hull_area,

            "side_solidity":
                solidity,

            "side_perimeter":
                perimeter,

            "side_bbox_width":
                w,

            "side_bbox_height":
                h,

            "side_aspect_ratio":
                aspect_ratio,

            "side_extent":
                extent,

            "side_major_axis_length":
                major_axis,

            "side_minor_axis_length":
                minor_axis,

            "side_eccentricity":
                eccentricity,

            "side_circularity":
                circularity
        }

    # ---------------------------------------------------
    # REAR FEATURES
    # ---------------------------------------------------

    else:

        rear_height_width_ratio = (
            h / (w + 1e-6)
        )

        return {

            # -----------------------------
            # REAR RATIO
            # -----------------------------

            "rear_height_width_ratio":
                rear_height_width_ratio,

            # -----------------------------
            # SHAPE FEATURES
            # -----------------------------

            "rear_mask_area":
                area,

            "rear_convex_hull_area":
                hull_area,

            "rear_solidity":
                solidity,

            "rear_perimeter":
                perimeter,

            "rear_bbox_width":
                w,

            "rear_bbox_height":
                h,

            "rear_aspect_ratio":
                aspect_ratio,

            "rear_extent":
                extent,

            "rear_major_axis_length":
                major_axis,

            "rear_minor_axis_length":
                minor_axis,

            "rear_eccentricity":
                eccentricity,

            "rear_circularity":
                circularity
        }