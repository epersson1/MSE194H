from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import random
import math

from gen_xrd import get_pattern

struc = mg.core.Structure.from_file('MnO.cif')
min_angle, max_angle = 10.0, 80.0
wavelength=1.5406
Q_min, Q_max = 4 * math.pi * np.sin(min_angle/2) / wavelength, 4 * math.pi * np.sin(max_angle/2) / wavelength
x = np.linspace(min_angle, max_angle, 4501)

y = get_pattern(struc, min_angle, max_angle)



#Q = (4 * pi * sin(theta))/lambda

#"When properly normalized, the inetnsity yields the total scattering function S(Q)"

# S(Q) = I(Q) / <b>^2
# S(Q) = I_coh(Q) - sum c_i|b_i|^2 / |c_ib_i|^2 + 1

# <b> = 1/ N sum b_v  = sum c_alpha * b_alpha
# c_alpha is atomic fraction
# b_alpha is scattering amplitude of the element alpha


def pdf(r):
    """Returns the Pair Distribution Function evaluated at distance r"""
    def func(Q):
        return Q * (S(Q) - 1) * np.sin(Q*r)
    return 2 / math.pi * quad(func, Q_min, Q_max)
