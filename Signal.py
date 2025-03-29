import numpy as np

# Supposons que Tf et N sont définis
Tf = 1.2  # Durée totale
N = 1000  # Nombre de valeurs

temps = np.linspace(0, Tf, N, endpoint=False)
print(temps)

import numpy as np

def chirp(temps, f0, Deltafe, T, E0):
    signal = np.zeros_like(temps)
    fe_t = f0 + (temps / T) * Deltafe
    mask = (temps >= 0) & (temps <= T)
    signal[mask] = E0 * np.sin(2 * np.pi * fe_t[mask] * temps[mask])
    return signal

# Exemple d'utilisation:
f0 = 200  # Fréquence porteuse en Hz
Deltafe = 100  # Bande de fréquences de modulation en Hz
T = 0.5  # Durée du signal en s
E0 = 1  # Amplitude du signal

signal_chirp = chirp(temps, f0, Deltafe, T, E0)

import matplotlib.pyplot as plt

plt.plot(temps, signal_chirp)
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Signal CHIRP')
plt.show()