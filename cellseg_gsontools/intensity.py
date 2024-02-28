from typing import Dict, List

import numpy as np
import scipy.ndimage as ndi
import skimage.filters as filters
from skimage import feature
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog
from skimage.util import invert

__all__ = ["intensity_props"]


def intensity_props(im: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes intensity properties of an image mask.

    Parameters:
        im (np.ndarray):
            Image to compute properties from. Shape (H, W).
        mask (np.ndarray):
            Mask to apply to the image. Shape (H, W). Can be a label image.

    Returns:
        Dict[str, np.ndarray]:
            Dictionary with the following properties:
            - area
            - mean_intensity
            - std_intensity
            - max_intensity
            - min_intensity
            - median_intensity
    """
    intensity = rgb2gray(im)

    objs = ndi.find_objects(mask)
    res = {
        "area": np.zeros(len(objs)),
        "mean_intensity": np.zeros(len(objs)),
        "median_intensity": np.zeros(len(objs)),
        "std_intensity": np.zeros(len(objs)),
        "max_intensity": np.zeros(len(objs)),
        "min_intensity": np.zeros(len(objs)),
    }

    for i, sl in enumerate(objs):
        image = mask[sl] == i + 1
        _intensity = intensity[sl] * image
        res["area"][i] = np.sum(image)
        res["mean_intensity"][i] = np.mean(_intensity[image], axis=0)
        res["median_intensity"][i] = np.median(_intensity[image], axis=0)
        res["std_intensity"][i] = np.std(_intensity[image], axis=0)
        res["max_intensity"][i] = np.max(_intensity[image], axis=0)
        res["min_intensity"][i] = np.min(_intensity[image], axis=0)

    return res


def chromatin_feats(
    im: np.ndarray,
    mask: np.ndarray = None,
    invert_image: bool = False,
    features: List[str] = ["mask"],
) -> np.ndarray:
    """Computes chromatin features from a given nuclei image.

    Three features are available:
    - mask: Thresholded image.
    - blobs: Detected blobs.
    - edges: Detected edges.

    Thresholds an image based on chromatin content using the yen method.

    Parameters:
        im (np.ndarray):
            Image to threshold. Shape (H, W, 3).
        mask (np.ndarray):
            Mask to apply to the image. Shape (H, W).
        invert_image (bool):
            Invert the image before thresholding. This is useful for chromatin clumps.
            If not used, the lighter regions will be thresholded.
        features (List[str]):
            Allowed: "mask", "blobs", "edges"

    Returns:
        mask (np.ndarray):
             Thresholded image i.e. a binary mask. Shape (H, W).
        blobs (np.ndarray):
            Detected blobs. Shape (N, 3). First two columns are the coordinates.
            The third column is the radius.
        edges (np.ndarray):
            Detected edges i.e. a binary mask. Shape (H, W).
    """
    allowed_features = ["mask", "blobs", "edges"]
    if not all([f in allowed_features for f in features]):
        raise ValueError("Invalid feature. Allowed: 'mask', 'blobs', 'edges'")

    # invert to get chromatin clumps
    if invert_image:
        im = invert(im)

    # convert rgb to gray
    im = rescale_intensity(im)
    im = rgb2gray(im)

    # apply mask
    if mask is not None:
        im *= mask

    # apply threshold
    res = {}

    if "mask" in features:
        res["mask"] = im > filters.threshold_yen(im, nbins=2000)

    if "blobs" in features:
        blobs = blob_dog(im, max_sigma=30, threshold=0.1)
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        res["blobs"] = blobs

    if "edges" in features:
        res["edges"] = feature.canny(im, sigma=3)

    return res
