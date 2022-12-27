import numpy as np


def get_tils(inflam_cells, distribution):
    """Get TIL cells by ranking cells bases on the distribution.

    Args:
    ---------
        inflam_cells: array of inflammatory cells
        distribution: 2D-distribution

    Returns:
    ---------
        Np.array of inflammatory cells with highest probability adjusted to amount of
        cells
    """
    TIL = []
    for i in range(0, len(inflam_cells["geometry"])):
        point = inflam_cells["geometry"].iloc[i].coords[:][0]
        TIL.append((i, distribution[int(point[0])][int(point[1])]))
    # TIL.sort(key = lambda x: x[1], reverse=True)
    wt = sum(x[1] for x in TIL)
    mask = [item[0] for item in TIL if item[1] * len(inflam_cells) / wt > 1]
    return np.take(np.array(list(zip(*TIL))[0]), mask)
