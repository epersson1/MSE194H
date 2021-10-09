from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import pandas as pd
import random
import math

from gen_xrd import get_pattern


def b(q):
    ions = ['Mn2+', 'O1-']
    q = 5
    form_factors = pd.read_csv('atomic_form_factors.csv')
    vals = []
    for i in ions:
        row = form_factors[form_factors['Element']==i]
        if not row.shape[0]:
            print("The values for calculating the atomic form factor of ", i, " are not in the table.")
            continue
        a = row[['a1', 'a2','a3','a4']]
        b = row[['b1', 'b2','b3','b4']]
        c = row['c'].iloc[0]
        val = sum(np.array(a.iloc[0,:]) * np.exp(np.array((-b * (q / (4*math.pi))**2).iloc[0,:]))) + c
        vals.append(val)
    return np.mean(vals)


def I(Q, data):
    angles = np.array(data['2theta'])

    q_angles = 4 * math.pi * np.sin(np.deg2rad(angles)) / wavelength
    difference = np.abs(q_angles-angles)
    closest_data = angles[np.argmin(np.abs(q_angles-angles))]
    return  data[data['2theta'] == closest_data]['Intensity'].iloc[0]

def S(Q, data):
    return I(Q, data) / b(Q)**2

def F(Q, data):
    return Q * (S(Q, data) - 1)

def pdf(r, data):
    """Returns the Pair Distribution Function evaluated at distance r"""
    Q_vals = (4 * math.pi * np.sin(np.deg2rad(angles / 2))) / wavelength
    Q_min, Q_max = min(Q_vals), max(Q_vals)
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = integrate.quad(lambda q: F(q, data) * np.sin(q*val), Q_min, Q_max)
        res[i] = y
    return res

def plot_pdf(r_min, r_max, data):
    plt.figure()
    r = np.linspace(r_min, r_max, 100)
    plt.plot(r, pdf(r, data))

    plt.xlabel(r'r', fontsize=16, labelpad=10)
    plt.ylabel('Density', fontsize=16, labelpad=12)

    plt.tight_layout()
    plt.show()

struc = mg.core.Structure.from_file('MnO.cif')
min_angle, max_angle = 10.0, 80.0
wavelength = 1.5406
calculator = xrd.XRDCalculator(wavelength=wavelength)
pattern = calculator.get_pattern(struc, two_theta_range=(min_angle, max_angle))
angles, intensities = pattern.x, pattern.y
data = pd.DataFrame(list(zip(angles, intensities)), columns=['2theta', 'Intensity'])

plot_pdf(0, 5, data)
