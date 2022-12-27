from base import gauss2d
from helper import *
from gradient import *

import numpy as np
import matplotlib.pyplot as plt

import timeit
starttime = timeit.default_timer()

cell_path = ""
slide_path = ""

slide = import_slide(slide_path)
W, H = slide.shape[0:2]

cells = get_cells(prep_json(import_json(cell_path)), get_size(slide_path))
neoplastic_cells = cells[cells["class_name"] == "neoplastic"]

points = []
for i in range(0, len(neoplastic_cells["geometry"])):
    point = neoplastic_cells["geometry"].iloc[i].coords[:][0]
    points.append(point)

z = gauss2d(W, H, points)
print("The time difference is :", timeit.default_timer() - starttime)
