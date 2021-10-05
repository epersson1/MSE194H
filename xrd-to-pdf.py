from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import random
import math

from gen_xrd import get_pattern

struc = mg.core.Structure.from_file('MnO.cif')
min_angle, max_angle = 10.0, 80.0
x = np.linspace(min_angle, max_angle, 4501)
y = get_pattern(struc, min_angle, max_angle)



#Q = (4 * pi * sin(theta))/lambda

#"When properly normalized, the inetnsity yields the total scattering function S(Q)"

# S(Q) = I(Q) / <b>^2

# <b> = 1/ N sum b_v  = sum c_alpha * b_alpha
# c_alpha is atomic fraction
# b_alpha is scattering amplitude of the element alpha


def pdf(r):
    """Returns the Pair Distribution Function evaluated at distance r"""
    return 2 / math.pi
