"""Génération et analyse des signaux sonar (Q1-Q4 du sujet)."""
import numpy as np
from scipy.signal import stft

C_EAU = 1500.0  # vitesse du son dans l'eau (m/s)


def temps_vecteur(Tf, N):
    """Q1 : N instants uniformément répartis dans [0, Tf[."""
    return np.linspace(0, Tf, N, endpoint=False)


def chirp(temps, f0, Deltafe, T, E0):
    """Q2 : signal CHIRP e(t) = E0 sin(2*pi*fe(t)*t) sur [0,T], nul ensuite.

    fe(t) = f0 + (t/T)*Deltafe (formule de l'énoncé). Attention : la fréquence instantanée
    vaut f0 + 2*Deltafe*t/T ; elle doit rester sous Nyquist (fs/2).
    """
    temps = np.asarray(temps, dtype=float)
    fe = f0 + temps / T * Deltafe
    return np.where((temps >= 0) & (temps <= T), E0 * np.sin(2 * np.pi * fe * temps), 0.0)


def simuler_echo(nature, temps, chirp_emis, distance=10.0, snr_db=10.0, rng=None):
    """Simule le signal de retour après réflexion sur un cylindre.

    nature : 'M' / 1 (métal : résonances longues, peu amorti)
             'R' / 0 (roche : absorption des hautes fréquences, écho court).
    Le retard vaut 2 d / c, l'amplitude décroît avec la distance, du bruit
    gaussien est ajouté selon le rapport signal/bruit snr_db.
    """
    rng = np.random.default_rng(rng)
    est_metal = nature in ("M", 1, 1.0)
    n = len(temps)
    te = temps[1] - temps[0]
    fs = 1 / te
    retard = 2 * distance / C_EAU
    freqs = np.fft.rfftfreq(n, te)

    # propagation : retard pur (déphasage linéaire) + atténuation géométrique
    spectre = np.fft.rfft(chirp_emis) * np.exp(-2j * np.pi * freqs * retard) / (1 + distance)

    if est_metal:  # légère coloration + gain fort
        gain = 1.0 + 0.3 * np.cos(2 * np.pi * freqs * rng.uniform(0.002, 0.006))
    else:          # passe-bas : la roche absorbe les hautes fréquences
        gain = 0.45 * np.exp(-freqs / rng.uniform(600, 1200))
    echo = np.fft.irfft(spectre * gain * rng.uniform(0.7, 1.3), n)

    if est_metal:  # résonances amorties du cylindre (dépendent de l'incidence)
        t_loc = np.clip(temps - retard, 0, None)
        fmax = fs / 2
        for _ in range(rng.integers(2, 5)):
            fr = rng.uniform(0.15, 0.85) * fmax
            tau = rng.uniform(0.03, 0.12)
            echo += (rng.uniform(0.1, 0.35) / (1 + distance)
                     * np.exp(-t_loc / tau) * np.sin(2 * np.pi * fr * t_loc) * (temps >= retard))

    puissance = np.mean(echo ** 2)
    sigma = np.sqrt(puissance / 10 ** (snr_db / 10)) if puissance > 0 else 0.0
    return echo + rng.normal(0, sigma, n)


def spectrogramme(s, Tf, n=128, fenetre="hamming"):
    """Q3 : STFT à 50 % de recouvrement, fenêtre de Hamming.

    Retourne f, t, |S|. Avec noverlap = n//2 : len(f) = n//2 + 1 et
    S.shape = (n//2 + 1, ~N//(n//2) + 1) (bords compris).
    """
    s = np.asarray(s, dtype=float)
    te = Tf / len(s)
    f, t, S = stft(s, fs=1 / te, window=fenetre, nperseg=n, noverlap=n // 2)
    return f, t, np.abs(S)
