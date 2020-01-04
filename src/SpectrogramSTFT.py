from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
from spectra_parser import SpectraParser

sp = SpectraParser("../dataset/Mineral_Absorbances.json",read_complete=True)
sp.output_file = "../dataset/properties.csv"

sp.input_spectra = os.path.join("../dataset/sources", "reflectances.csv")
x = sp.x()
[f, t, Zxx] = signal.stft(x[0], fs=1, nperseg=100, noverlap=50)
plt.pcolormesh(t, f, np.log(np.abs(Zxx)))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()