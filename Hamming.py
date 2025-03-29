import numpy as np
from scipy.signal import stft

# Données d'exemple
N = 1000  # Taille du signal
Tf = 1.2  # Durée totale
s = np.random.randn(N)  # Signal numérique d'exemple

# Période d'échantillonnage
te = Tf / N

# Paramètres pour la transformation de Fourier locale
n = 128  # Nombre de valeurs autour de chaque instant τ
overlap = n // 2  # Recouvrement de 50%

# Calcul de la STFT (Short-Time Fourier Transform)
f, t, S = stft(s, fs=1/te, window='hamming', nperseg=n, noverlap=overlap)

# Taille des grandeurs obtenues
nf = n // 2 + 1  # Nombre de fréquences
nt = (N - overlap) // (n - overlap)  # Nombre d'instants τ retenus

print(f"Taille de f: {len(f)} (devrait être {nf})")
print(f"Taille de t: {len(t)} (devrait être {nt})")
print(f"Taille de S: {S.shape} (devrait être ({nf}, {nt}))")