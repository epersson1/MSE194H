from diffpy.srreal.pdfcalculator import ConstantPeakWidth
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.Structure import Structure
import matplotlib.pyplot as plt


struc = Structure(filename='MnO.cif')
struc.Uisoequiv = 0.005

pdfc = PDFCalculator()
x, y = pdfc(struc)

plt.figure()

plt.plot(x, y)

plt.xlabel(r'r ($\mathrm{\AA}$)', fontsize=14, labelpad=8)
plt.ylabel('g(r)', fontsize=14, labelpad=10)

plt.tight_layout()
plt.show()
