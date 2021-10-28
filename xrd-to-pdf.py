from pymatgen.analysis.diffraction import xrd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import pandas as pd
import json
import random
import math

from gen_xrd import get_pattern
import re


# THE DATA - SIMULATED
# struc = mg.core.Structure.from_file('MnO.cif')
#
# min_angle, max_angle = 10.0, 80.0
# wavelength = 1.5406
# calculator = xrd.XRDCalculator(wavelength=wavelength)
# pattern = calculator.get_pattern(struc, two_theta_range=(min_angle, max_angle), scaled=False)
# two_angles, intensities = pattern.x, pattern.y



# THE DATA - EXPERIMENTAL
structure = 'MnO'
path = 'Experimental/Clean/'
two_angles, intensities = [], []
with open(path + structure + '.xy', 'r') as f:
    contents = f.readlines()
    for line in contents:
        if 'Wavelength' in line:
            wavelength = float(line.split("=")[1].split(" ")[1])
        else:
            vals = line.split("\t")
            if len(vals) == 2:
                two_angles.append(float(vals[0]))
                intensities.append(float(vals[1].strip('\n')))
            else:
                print(vals[0])
two_angles = np.array(two_angles)
intensities = np.array(intensities)


# Processing
Q_vals = (4 * math.pi * np.sin(np.deg2rad(two_angles/2))) / wavelength
Q_min, Q_max = min(Q_vals), max(Q_vals)



def I(Q):
    """Return value if Q has been observed, else 0"""
    if Q in Q_vals:
        return intensities[np.argmin(np.abs(Q - Q_vals))]
    return 0

# THE THINGS YOU CALCULATE FROM THE MATERIAL
def scattering_factors():
    ions = re.findall('[A-Z][^A-Z]*', structure)
    form_factors = pd.read_csv('atomic_form_factors.csv')
    factors = []
    for i in ions:
        row = form_factors[form_factors['Element']==i]
        if not row.shape[0]:
            print("The values for calculating the atomic form factor of ", i, " are not in the table.")
            continue
        a = np.array(row[['a1', 'a2','a3','a4']].iloc[0])
        b = np.array(row[['b1', 'b2','b3','b4']].iloc[0])
        c = row['c'].iloc[0]
        factors.append((a,b,c))
    return dict(zip(ions, factors))
factors = scattering_factors()

def b(q):
    """Returns the average scattering amplitude of the material"""
    vals = []
    for tup in factors.values():
        a,b,c = tup
        val = sum(a * np.exp(-b * (q/(4 * math.pi))**2)) + c
        vals.append(val)
    return np.mean(vals)


# THE MATH
def S(Q):
    return I(Q) / (b(Q)**2)

def F(Q):
    return Q * (S(Q) - 1)

def pdf(r):
    """Returns the Pair Distribution Function evaluated at distance r"""
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = integrate.quad(lambda q: F(q) * np.sin(q*val), Q_min, Q_max)
        res[i] = 2 / math.pi * y
    return res


# THE PLOTTING
def plot_pdf(r_min, r_max):
    plt.figure()
    r = np.linspace(r_min, r_max, 1000)
    plt.plot(r, pdf(r))

    plt.xlabel(r'r (Å)', fontsize=16, labelpad=10)
    plt.ylabel('g(r)', fontsize=16, labelpad=12)

    plt.tight_layout()
    plt.show()

plot_pdf(0, 10)
