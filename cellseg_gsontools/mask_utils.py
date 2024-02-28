from typing import List, Tuple

import numpy as np
from skimage.morphology import dilation, disk, erosion

__all__ = ["bounding_box", "crop_to_bbox", "maskout_array"]


def bounding_box(mask: np.ndarray) -> List[int]:
    """Bounding box coordinates for an instance that is given as input.

    This assumes that the `inst_map` has only one instance in it.

    Parameters:
        inst_map (np.ndarray):
            Instance labelled mask. Shape (H, W).

    Returns:
        List[int]:
            List of the origin- and end-point coordinates of the bbox.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmax += 1
    cmax += 1

    return [rmin, rmax, cmin, cmax]


def crop_to_bbox(
    src: np.ndarray, mask: np.ndarray, dilation_level: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops an image and mask to the bounding box of the mask.

    Parameters:
        src (np.ndarray):
            Source image. Shape (H, W, 3).
        mask (np.ndarray):
            Mask to crop the image with. Shape (H, W).
        dilation_level (int):
            Dilation level for the mask.

    Raises:
        ValueError: If the src array is not 2D or 3D.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Cropped image and mask.
    """
    if not 2 <= src.ndim <= 3:
        raise ValueError("src must be a 2D or 3D array.")

    if dilation_level > 0:
        mask = dilation(mask, disk(dilation_level))

    ymin, ymax, xmin, xmax = bounding_box(mask)

    # erode back to orig mask
    if dilation_level > 0:
        mask = erosion(mask, disk(dilation_level))

    mask = mask[ymin:ymax, xmin:xmax]
    src = src[ymin:ymax, xmin:xmax]

    return src, mask


def maskout_array(
    src: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Masks out the input array with the given mask."""
    if not 2 <= src.ndim <= 3:
        raise ValueError("src must be a 2D or 3D array.")

    if src.ndim == 3:
        src = src * mask[..., None]
    else:
        src = src * mask

    return src
