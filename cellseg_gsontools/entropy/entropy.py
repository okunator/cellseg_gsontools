import geopandas as gpd
import numpy as np
from helper import get_points
from sklearn.cluster import DBSCAN


def array_entropy(array: np.array, bins: np.linspace) -> float:
    """Calculate Shannon entropy for array.

    Args:
    ---------
        array: numerical array
        bins: numerical array

    Returns:
    ---------
        float: entropy
    """
    digitized = np.histogram(array, bins)
    array = digitized[0]

    samples_probability = [float(h) / np.sum(array) for h in array]
    return -np.sum([p * np.log(p) for p in samples_probability if p != 0])


def image_entropy(img: np.array) -> float:
    """Calculate entropy for a matrix.

    Args:
    ---------
        array: numerical array image

    Returns:
    ---------
        float: entropy
    """
    histogram, bin_edges = np.histogram(img)
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -np.sum([p * np.log(p) for p in samples_probability if p != 0])


def image_entropy_results(img: np.array, k: int) -> np.array:
    """Apply kxk entropy convolution to image matrix.

    Args:
    ---------
        array: numerical array
            k: size of convolution kernel

    Returns:
    ---------
        float: entropy
    """
    results = np.zeros([len(img), len(img[0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            sample = img[i : i + k, j : j + k]
            entropy = image_entropy(sample)
            results[i, j] = entropy
    return results


def image_smooth(img: np.array, k: int) -> np.array:
    """Apply kxk smoothing kernel to matrix.

    Args:
    ---------
        array: numerical array
            k: size of convolution kernel

    Returns:
    ---------
        numerical array
    """
    results = np.zeros([len(img), len(img[0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            sample = img[i : i + k, j : j + k]
            results[i, j] = np.mean(sample)
    return results


def image_sharpen(img: np.array, k: int) -> np.array:
    """Apply kxk sharpening kernel to matrix.

    Args:
    ---------
        array: numerical array
            k: size of convolution kernel

    Returns:
    ---------
        numerical array
    """
    results = np.zeros([len(img), len(img[0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            sample = img[i : i + k, j : j + k]
            results[i, j] = np.max(sample)
    return results


def image_min(img: np.array, k: int) -> np.array:
    """Apply kxk minimum kernel to matrix.

    Args:
    ---------
        array: numerical array
            k: size of convolution kernel

    Returns:
    ---------
        numerical array
    """
    results = np.zeros([len(img), len(img[0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            sample = img[i : i + k, j : j + k]
            results[i, j] = np.min(sample)
    return results


def add_cell_shape(cells: gpd.geodataframe) -> float:
    """Add cell shape metric to cell table.

    Args:
    ---------
        Geodataframe: array of cells

    Returns:
    ---------
        Geodataframe: array of cells
    """
    cell = list(cells.iterrows())

    cells["area"] = [c[1]["geometry"].area for c in cell]
    cells["perimeter"] = [c[1]["geometry"].length for c in cell]
    cells["shape"] = np.sqrt(cells["area"]) / cells["perimeter"]

    return cells


def cell_shape_metric(cells: gpd.geodataframe) -> float:
    """Calculate entropy of cell shapes.

    Args:
    ---------
        Geodataframe: array of cells

    Returns:
    ---------
        float: entropy of cell shapes in table
    """
    return array_entropy(cells)


def lesion_segment(cells: gpd.geodataframe, points, eps=200):
    """Cluster lesions.

    Args:
    ---------
        cells: Geodataframe of cells
       points: Coordinates of cells

    Returns:
    ---------
        Geodataframe of cells
    """
    db = DBSCAN(eps=eps, min_samples=0).fit(points, cells["weight"])
    labels = db.labels_

    cells["label"] = labels
    return cells


def spatial_entropy(cells: gpd.geodataframe):
    """Calculate entropy of marginal coordinate distributions of cells.

    Args:
    ---------
        cells: Geodataframe of cells

    Returns:
    ---------
        float: entropy
    """
    xs = [p[0] for p in get_points(cells)]
    ys = [p[1] for p in get_points(cells)]

    X = np.histogram(xs, 20)
    Y = np.histogram(ys, 20)

    samples_probability = Y[0] / np.sum(Y[0])
    a = -np.sum([p * np.log(p) for p in samples_probability if p != 0])

    samples_probability = X[0] / np.sum(X[0])
    b = -np.sum([p * np.log(p) for p in samples_probability if p != 0])

    return (a, b)
