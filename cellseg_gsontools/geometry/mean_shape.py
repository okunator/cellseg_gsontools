import geopandas as gpd
import numpy as np

try:
    import geomstats.backend as gs
    from geomstats.geometry.discrete_curves import DiscreteCurves, SRVMetric
    from geomstats.geometry.euclidean import Euclidean
    from geomstats.geometry.pre_shape import PreShapeSpace
    from geomstats.learning.frechet_mean import FrechetMean
except ImportError:
    raise ImportError(
        "`geomstats` package required. Install from github: "
        """git clone https://github.com/geomstats/geomstats.git
           cd geomstats
           pip3 install ."""
    )


M_AMBIENT = 2
r2 = Euclidean(dim=2)
curves_r2 = DiscreteCurves(ambient_manifold=r2)
k_sampling_points = 50
PRESHAPE_SPACE = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=k_sampling_points)
PRESHAPE_METRIC = PRESHAPE_SPACE.embedding_space.metric


def center_curves(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Prepare curves by centering them.

    Parameters
    ----------
    df : gpd.geodataframe
        Dataframe with curves to be centered.

    Returns
    -------
    gpd.geodataframe
    """
    curves = []

    for curve in df["geometry"]:
        x, y = curve.exterior.coords.xy
        x = x - np.median(x)
        y = y - np.median(y)

        centered = list(zip(x, y))
        curves.append(centered)

    curves = np.array(curves)

    df["geometry"] = curves

    return df


def interpolate(contours, n: int = k_sampling_points):
    """Interpolate curve to given number of points.

    Parameters
    ----------
    curve : np.array
        Curve to be interpolated.
    n : np.array
        Number of points for wanted curve.

    Returns
    -------
    np.array
    """
    xc = [a[0] for a in contours]
    yc = [a[1] for a in contours]

    # spacing of x and y points.
    dy = np.diff(yc)
    dx = np.diff(xc)

    # distances between consecutive coordinates
    dS = np.sqrt(dx**2 + dy**2)
    dS = np.append(np.zeros(1), dS)  # include starting point

    # Arc length and perimeter
    d = np.cumsum(dS)
    perim = d[-1]

    # divide the perimeter to evenly spaced values
    ds = perim / n
    dSi = np.arange(0, n) * ds
    dSi[-1] = dSi[-1] - 5e-3

    # interpolate the x and y coordinates
    yi = np.interp(dSi, d, yc)
    xi = np.interp(dSi, d, xc)

    return np.array(list(zip(xi, yi)))


def apply_func_to_ds(input_ds, func):
    """Apply the input function func to the input dictionnary input_ds.

    This function goes through the dictionnary structure and applies
    func to every cell in input_ds[treatment][line].

    It stores the result in a dictionnary output_ds that is returned
    to the user.

    Parameters
    ----------
    input_ds : dict
        Input dictionnary, with keys treatment-line.
    func : callable
        Function to be applied to the values of the dictionnary, i.e.
        the cells.

    Returns
    -------
    output_ds : dict
        Output dictionnary, with the same keys as input_ds.
    """
    output_ds = {}
    output_list = []
    for one_cell in input_ds:
        output_list.append(func(one_cell))
    output_ds = gs.array(output_list)
    return output_ds


def exhaustive_align(curve, base_curve):
    """Align given curve with base curve, orientation minimizing geodesic distance.

    Parameters
    ----------
    curve : np.array
        Curve to be aligned.
    base_curve : np.array
        Curve to be aligned to.

    Returns
    -------
    aligned_curve: gs.array
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = PRESHAPE_METRIC.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def frechet_mean(df: gpd.GeoDataFrame, metric=curves_r2.srv_metric):
    """Calculate the Frechet mean in a give metric on an array of curves.

    Parameters
    ----------
    df : gpd.geoDataframe
        Input dictionnary, with keys treatment-line.
    metric : callable
        Function to be applied to the values of the dictionnary, i.e.
        the cells.

    Returns
    -------
    np.array
    """
    ds_proj = [interpolate(curve) for curve in center_curves(df)["geometry"]]

    BASE_CURVE = ds_proj[0]
    ds_align = apply_func_to_ds(ds_proj, func=lambda x: exhaustive_align(x, BASE_CURVE))

    mean = FrechetMean(metric=metric, method="default")
    mean.fit(ds_align)

    mean_estimate = mean.estimate_

    mean_estimate_clean = mean_estimate[~gs.isnan(gs.sum(mean_estimate, axis=1)), :]
    mean_estimate_clean = interpolate(mean_estimate_clean, n=k_sampling_points)

    return mean_estimate_clean


def dists_to_mean(curves, global_mean):
    """Calculate every cells geodesic distance to mean shape of cell collection.

    Parameters
    ----------
    curves : np.array
        Collecion of curves, Prepped and aligned.
    global_mean : np.array
       Calculated Frechet mean of curve collection.

    Returns
    -------
    np.array
    """
    return [SRVMetric.dist(curve, global_mean) for curve in curves]
