"""Fixed, label-free features at native scale. No learned image preprocessing."""
from pathlib import Path
import cv2
import numpy as np

VERSION = 'native-quality-v1'
BASIC = ['laplacian_raw', 'laplacian_smooth', 'tenengrad_smooth', 'local_tenengrad_p75']


def tiles(a):
    return [t for band in np.array_split(a, 4) for t in np.array_split(band, 4, axis=1)]


def gradient(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, gx*gx + gy*gy


def extract(path, basic_only=False):
    raw = np.frombuffer(Path(path).read_bytes(), np.uint8)
    gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (768, 1024):
        raise ValueError(f'Expected native 1024x768 image: {path}')
    return from_gray(gray, basic_only=basic_only)


def from_gray(gray, basic_only=False):
    a = gray.astype(np.float32) / 255.0
    s = cv2.GaussianBlur(a, (0, 0), 1.0)
    gx, gy, energy = gradient(s)
    local = np.array([float(t.mean()) for t in tiles(energy)])
    f = dict(laplacian_raw=float(cv2.Laplacian(a, cv2.CV_32F, ksize=1).var()),
             laplacian_smooth=float(cv2.Laplacian(s, cv2.CV_32F, ksize=1).var()),
             tenengrad_smooth=float(energy.mean()), local_tenengrad_p75=float(np.percentile(local, 75)))
    if basic_only:
        return f
    for p in (10, 25, 50, 90):
        f[f'local_tenengrad_p{p}'] = float(np.percentile(local, p))
    f['local_tenengrad_min'] = float(local.min())
    f['focus_low_high_ratio'] = float(np.percentile(local, 10) / (np.percentile(local, 90) + 1e-12))
    f['focus_cv'] = float(local.std() / (local.mean() + 1e-12))
    f['low_focus_tile_fraction'] = float((local < .25 * np.median(local)).mean())
    for sigma in (2, 4):
        sm = cv2.GaussianBlur(a, (0, 0), sigma)
        _, _, en = gradient(sm)
        lo = [float(t.mean()) for t in tiles(en)]
        f[f'tenengrad_sigma{sigma}'] = float(en.mean())
        for p in (25, 75):
            f[f'local_sigma{sigma}_p{p}'] = float(np.percentile(lo, p))
        f[f'laplacian_sigma{sigma}'] = float(cv2.Laplacian(sm, cv2.CV_32F, ksize=1).var())
    f['fine_coarse_energy_ratio'] = f['tenengrad_smooth'] / (f['tenengrad_sigma4'] + 1e-12)
    diag = (a[:-1, :-1] - a[1:, :-1] - a[:-1, 1:] + a[1:, 1:]) / 2
    f['noise_diagonal_mad'] = float(np.median(np.abs(diag - np.median(diag))) / .67448975)
    f['highpass_std'] = float((a-s).std())
    f['noise_signal_ratio'] = f['noise_diagonal_mad'] / (float(s.std()) + 1e-6)
    f['mean_brightness'] = float(a.mean())
    f['contrast_std'] = float(s.std())
    f['contrast_p95_p05'] = float(np.percentile(s, 95) - np.percentile(s, 5))
    f['dark_fraction'] = float((a <= .02).mean())
    f['bright_fraction'] = float((a >= .98).mean())
    glare = (a >= .98).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(glare, connectivity=8)
    f['largest_glare_fraction'] = float(stats[1:, cv2.CC_STAT_AREA].max()/a.size) if n > 1 else 0.0
    means = np.array([t.mean() for t in tiles(s)])
    contrasts = np.array([t.std() for t in tiles(s)])
    f['illumination_range'] = float(means.max()-means.min())
    f['illumination_cv'] = float(means.std()/(means.mean()+1e-6))
    for p in (10, 50, 90):
        f[f'local_contrast_p{p}'] = float(np.percentile(contrasts, p))
    normalized = local / (contrasts**2 + 1e-6)
    for p in (10, 50, 75):
        f[f'contrast_normalized_focus_p{p}'] = float(np.percentile(normalized, p))
    for p in (50, 90, 99):
        f[f'gradient_energy_p{p}'] = float(np.percentile(energy, p))
    coherence = []
    for tx, ty in zip(tiles(gx), tiles(gy)):
        xx, yy, xy = float((tx*tx).mean()), float((ty*ty).mean()), float((tx*ty).mean())
        coherence.append(np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-12))
    f['orientation_coherence_mean'] = float(np.mean(coherence))
    f['orientation_coherence_p75'] = float(np.percentile(coherence, 75))
    assert all(np.isfinite(v) and v >= 0 for v in f.values())
    return f
