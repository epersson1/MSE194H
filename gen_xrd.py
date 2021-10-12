from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import random
import math


def calc_std_dev(two_theta, tau, wavelength):
    """
    Calculate standard deviation based on angle (two theta) and domain size (tau).
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.9 ## shape factor
    wavelength *= 0.1 ## angstrom to nm
    theta = np.radians(two_theta/2.) ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2


def get_pattern(struc, min_angle=10.0, max_angle=80.0, wavelength=1.5406):
    """
    Calculate a continuous XRD spectrum for a given structure.
    Args:
        struc: pymatgen Structure object
        min_angle: minimum two theta
        max_angle: maximum two theta
    Return:
        signal: XRD pattern (numpy array)
    """
    calculator = xrd.XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(struc, two_theta_range=(min_angle, max_angle))
    angles, intensities = pattern.x, pattern.y

    steps = np.linspace(min_angle, max_angle, 4501)

    signals = np.zeros([len(angles), steps.shape[0]])

    for i, ang in enumerate(angles):
        # Map angle to closest datapoint step
        idx = np.argmin(np.abs(ang-steps))
        signals[i,idx] = intensities[i]

    # Convolute every row with unique kernel
    # Iterate over rows; not vectorizable, changing kernel for every row
    domain_size = 25.0
    step_size = (max_angle - min_angle)/4501
    for i in range(signals.shape[0]):
        row = signals[i,:]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size, wavelength)
        # Gaussian kernel expects step size 1 -> adapt std_dev
        signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size, mode='constant')

    # Combine signals
    signal = np.sum(signals, axis=0)

    # Normalize signal
    signal = 100 * signal / max(signal)

    # Add some noise to make pattern realistic
    noise = np.random.normal(0, 0.1, 4501)
    signal = signal + noise

    return signal

def plot_pattern(x, y):
    plt.figure()

    plt.plot(x, y)

    plt.xlabel(r'2$\Theta$', fontsize=16, labelpad=10)
    plt.ylabel('Intensity', fontsize=16, labelpad=12)

    plt.tight_layout()
    plt.show()

struc = mg.core.Structure.from_file('MnO.cif')

min_angle, max_angle = 10.0, 80.0
x = np.linspace(min_angle, max_angle, 4501)
y = get_pattern(struc, min_angle, max_angle)

#plot_pattern(x, y)
