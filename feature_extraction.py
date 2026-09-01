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

    try:

        print(
            f"\n========== {view_type.upper()} "
            f"FEATURE EXTRACTION =========="
        )

        # ---------------------------------------------------
        # FIX IMAGE ORIENTATION
        # ---------------------------------------------------

        print("Step 1: Fixing image orientation...")

        image = ImageOps.exif_transpose(image)

        print(
            f"Image size: {image.size}"
        )

        # ---------------------------------------------------
        # SEGMENTATION
        # ---------------------------------------------------

        print(
            "Step 2: Running segmentation..."
        )

        sticker_mask, cattle_mask = \
            get_segmentation_masks(
                seg_model,
                image,
                view_type
            )

        print(
            "Segmentation function completed."
        )

        # ---------------------------------------------------
        # CHECK CATTLE MASK
        # ---------------------------------------------------

        if cattle_mask is None:

            print(
                "ERROR: Cattle mask is None."
            )

            return None

        cattle_mask = cattle_mask.astype(
            np.uint8
        )

        print(
            "Cattle mask obtained."
        )

        print(
            f"Cattle mask shape: "
            f"{cattle_mask.shape}"
        )

        print(
            f"Cattle mask pixels: "
            f"{int(np.sum(cattle_mask > 0))}"
        )

        # ---------------------------------------------------
        # SCALE
        # ---------------------------------------------------

        if view_type == "Side":

            print(
                "Step 3: Computing scale "
                "from reference sticker..."
            )

            if sticker_mask is None:

                print(
                    "ERROR: Sticker mask is None."
                )

                return None

            print(
                f"Sticker mask pixels: "
                f"{int(np.sum(sticker_mask > 0))}"
            )

            scale = compute_scale_from_sticker(
                sticker_mask
            )

            if scale is None:

                print(
                    "ERROR: Scale computation failed."
                )

                return None

            print(
                f"Computed scale = {scale}"
            )

        else:

            print(
                "Step 3: Rear view detected. "
                "Using scale = 1.0"
            )

            scale = 1.0

        # ---------------------------------------------------
        # FIND CONTOURS
        # ---------------------------------------------------

        print(
            "Step 4: Finding cattle contour..."
        )

        contours, _ = cv2.findContours(
            cattle_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(
            f"Number of contours found: "
            f"{len(contours)}"
        )

        if len(contours) == 0:

            print(
                "ERROR: No cattle contour found."
            )

            return None

        # Select largest contour

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        # ---------------------------------------------------
        # BASIC SHAPE FEATURES
        # ---------------------------------------------------

        print(
            "Step 5: Calculating shape features..."
        )

        x, y, w, h = cv2.boundingRect(
            cnt
        )

        area = cv2.contourArea(
            cnt
        )

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        hull = cv2.convexHull(
            cnt
        )

        hull_area = cv2.contourArea(
            hull
        )

        solidity = (
            area /
            (hull_area + 1e-6)
        )

        extent = (
            area /
            (w * h + 1e-6)
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

            major_axis = max(
                MA,
                ma
            )

            minor_axis = min(
                MA,
                ma
            )

        else:

            major_axis = max(
                w,
                h
            )

            minor_axis = min(
                w,
                h
            )

        eccentricity = np.sqrt(
            max(
                0,
                1 -
                (
                    (minor_axis ** 2) /
                    (major_axis ** 2 + 1e-6)
                )
            )
        )

        # ---------------------------------------------------
        # ASPECT RATIO
        # ---------------------------------------------------

        if view_type == "Side":

            # Same definition used during training:
            # side_aspect_ratio = width / height

            aspect_ratio = (
                w /
                (h + 1e-6)
            )

        else:

            # Same definition used during training:
            # rear_aspect_ratio = height / width

            aspect_ratio = (
                h /
                (w + 1e-6)
            )

        print(
            f"Bounding box: "
            f"x={x}, y={y}, w={w}, h={h}"
        )

        print(
            f"Area: {area}"
        )

        print(
            f"Perimeter: {perimeter}"
        )

        print(
            f"Convex hull area: {hull_area}"
        )

        # ---------------------------------------------------
        # KEYPOINT DETECTION
        # ---------------------------------------------------

        print(
            "Step 6: Running keypoint detection..."
        )

        kp = get_keypoints(
            kp_model,
            image
        )

        if kp is None:

            print(
                "ERROR: Keypoint detection "
                "returned None."
            )

            return None

        print(
            f"Keypoints shape: {kp.shape}"
        )

        print(
            f"Number of keypoints: {len(kp)}"
        )

        # ---------------------------------------------------
        # CHECK KEYPOINT COUNT
        # ---------------------------------------------------

        if view_type == "Side":

            required_keypoints = 9

        else:

            required_keypoints = 4

        if len(kp) < required_keypoints:

            print(
                f"ERROR: Expected at least "
                f"{required_keypoints} keypoints, "
                f"but received {len(kp)}."
            )

            return None

        # ---------------------------------------------------
        # SIDE FEATURES
        # ---------------------------------------------------

        if view_type == "Side":

            print(
                "Step 7: Calculating side-view "
                "linear features..."
            )

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

            print(
                f"Body length = "
                f"{body_length}"
            )

            print(
                f"Chest depth = "
                f"{chest_depth}"
            )

            print(
                f"Rear depth = "
                f"{rear_depth}"
            )

            print(
                f"Hip height = "
                f"{hip_height}"
            )

            print(
                f"Body oblique length = "
                f"{body_oblique_length}"
            )

            print(
                "Side-view feature extraction "
                "completed successfully."
            )

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

            print(
                "Step 7: Calculating rear-view "
                "features..."
            )

            rear_height_width_ratio = (
                h /
                (w + 1e-6)
            )

            print(
                f"Rear height/width ratio = "
                f"{rear_height_width_ratio}"
            )

            print(
                "Rear-view feature extraction "
                "completed successfully."
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

    # ---------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------

    except Exception as e:

        print(
            f"\nERROR in {view_type} "
            f"feature extraction:"
        )

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error message: "
            f"{str(e)}"
        )

        import traceback

        traceback.print_exc()

        return None