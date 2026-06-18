#!/usr/bin/env python3
"""
sem_interactive_v2.py
Improved version: for image recognition effect demonstration

Improvements:
  1. Phase 1-2: Display popup window using plt.show(), do not save locally
  2. Phase 3-4: Save only these two images (FFT + final result)
  3. Watershed replaced with Gradient Edge (maximum gradient magnitude recognition)
  4. Phase 2 auto-executed, no intermediate confirmation needed
"""

import sys
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # VSCode compatible interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage as ndi
from scipy.signal import find_peaks, savgol_filter

# ─────────────────────────────────────────────────────────
# Paths and global parameters
# ─────────────────────────────────────────────────────────

# Image path
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_PATH = SCRIPT_DIR / 'SEM_RAW.png'
OUT = SCRIPT_DIR / 'outputs'
OUT.mkdir(exist_ok=True)

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Panel f crop coordinates (1100°C)
PX, PY, PW, PH = 519, 412, 510, 255

# Instrument calibration
PPU        = 18.0
CORRECTION = 0.50

# Unified pore separation parameters
BORDER        = 3
MIN_SEED_PX   = 6
MIN_PORE_AREA = 10
PORE_PERC     = 8

# Greedy packing parameters
MIN_SEP_FRAC = 0.90
RADIUS_STEP  = 0.85
COVER_THRESH = 0.002
BRIGHT_W     = 0.3

# Visualization colors
PORE_RGB  = np.array([180,  60, 200], np.float32)   # Purple
BOUND_RGB = np.array([ 45,  82, 160], np.float32)   # Blue
SEED_RGB  = np.array([255, 220,  50], np.float32)   # Yellow

mpl.rc('font', family='sans-serif', size=12)


# ─────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────

def valid_mask(h, w):
    m = np.ones((h, w), bool)
    m[:BORDER, :] = m[-BORDER:, :] = m[:, :BORDER] = m[:, -BORDER:] = False
    return m


def _remove_small(mask, min_area):
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask)
    for lbl in range(1, n):
        if int((labels == lbl).sum()) >= min_area:
            out |= (labels == lbl)
    return out


def _adaptive_ksize(kept):
    dist_seed = ndi.distance_transform_edt(~kept)
    sv = dist_seed[kept]
    sv = sv[sv < 50]
    median_r = float(np.median(sv)) if len(sv) else 3.0
    return max(5, int(median_r * 3) | 1)


# ─────────────────────────────────────────────────────────
# Three pore separation schemes
# ─────────────────────────────────────────────────────────

def segment_dark(gray, valid):
    """Darkest 8% pixels as seeds"""
    px  = gray[valid]
    thr = np.percentile(px, PORE_PERC)
    seeds_raw = (gray <= thr) & valid
    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds = cv2.morphologyEx(seeds_raw.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    seeds &= valid
    kept = _remove_small(seeds, MIN_SEED_PX)
    k_size = _adaptive_ksize(kept)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_CLOSE, k_grow)
    filled = ndi.binary_fill_holes(closed.astype(bool))
    pore_mask = _remove_small(filled, MIN_PORE_AREA) & valid
    return pore_mask, kept.astype(bool), k_size


def segment_bright(gray, valid):
    """Brightest 8% pixels as seeds"""
    px  = gray[valid]
    thr = np.percentile(px, 100 - PORE_PERC)
    seeds_raw = (gray >= thr) & valid
    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds = cv2.morphologyEx(seeds_raw.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    seeds &= valid
    kept = _remove_small(seeds, MIN_SEED_PX)
    k_size = _adaptive_ksize(kept)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_CLOSE, k_grow)
    filled = ndi.binary_fill_holes(closed.astype(bool))
    pore_mask = _remove_small(filled, MIN_PORE_AREA) & valid
    return pore_mask, kept.astype(bool), k_size


def segment_gradient(gray, valid):
    """
    Gradient recognition: regions with maximum gradient magnitude as pore boundary
    (Alternative to Watershed, more stable for various SEM images)
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2).astype(np.float32)
    grad[~valid] = 0
    
    # Highest 8% gradient magnitude pixels as seeds (pore boundary)
    px = grad[valid]
    thr = np.percentile(px, 100 - PORE_PERC)
    seeds_raw = (grad >= thr) & valid
    
    # Morphological processing
    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds = cv2.morphologyEx(seeds_raw.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    seeds &= valid
    kept = _remove_small(seeds, MIN_SEED_PX)
    
    k_size = _adaptive_ksize(kept)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_CLOSE, k_grow)
    filled = ndi.binary_fill_holes(closed.astype(bool))
    pore_mask = _remove_small(filled, MIN_PORE_AREA) & valid
    return pore_mask, kept.astype(bool), k_size


# ─────────────────────────────────────────────────────────
# Phase 1: Display comparison (do not save)
# ─────────────────────────────────────────────────────────

def show_phase1(crop, results_dict):
    """Display popup window using plt.show(), do not save locally"""
    names  = ['Dark', 'Bright', 'Gradient']
    h, w   = crop.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        'Phase 1 — Pore Segmentation Comparison  |  1100 °C  |  Panel f',
        fontsize=13, fontweight='bold', y=1.00)

    for i, name in enumerate(names):
        pm, seeds, k = results_dict[name]
        n_pores = int(ndi.label(pm)[1])
        pore_pct = pm.sum() / (h * w) * 100

        # Top row: pore mask
        ov = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        ov[pm] = ov[pm] * 0.25 + PORE_RGB * 0.75
        axes[0, i].imshow(ov.clip(0, 255).astype(np.uint8))
        k_str = f'  k={k}' if k > 0 else '  (gradient)'
        axes[0, i].set_title(
            f'{name}{k_str}\nN={n_pores}   area={pore_pct:.1f}%',
            fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Bottom row: seeds
        ov2 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        ov2[seeds] = ov2[seeds] * 0.2 + SEED_RGB * 0.8
        n_seeds = int(ndi.label(seeds)[1])
        axes[1, i].imshow(ov2.clip(0, 255).astype(np.uint8))
        axes[1, i].set_title(f'Seeds  N={n_seeds}', fontsize=11)
        axes[1, i].axis('off')

    for row, label in enumerate(['Pore Mask', 'Seed Mask']):
        fig.text(0.005, 0.72 - row * 0.44, label,
                 va='center', rotation='vertical',
                 fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    plt.show()


# ─────────────────────────────────────────────────────────
# Phase 2-4: Pipeline
# ─────────────────────────────────────────────────────────

def make_boundary(gray, valid, pore_mask, min_contour_area=30):
    """
    Boundary detection: remove isolated lines (grain boundaries), 
    keep only closed contours (pore boundaries)
    
    Parameters:
      min_contour_area: Minimum contour area (px²), contours below this value are deleted
                       Recommended: 20-50 (adjust based on pore size)
    """
    k3           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pore_u8      = pore_mask.astype(np.uint8)
    dilated      = cv2.dilate(pore_u8, k3)
    pore_outline = (dilated - pore_u8).astype(bool) & valid
    
    # Canny edge detection
    px    = gray[valid]
    bt    = np.percentile(px, 97)
    canny = cv2.Canny((gray >= bt).astype(np.uint8) * 255, 50, 150)
    canny_bnd = (canny > 0) & valid
    
    # Contour area filtering: keep only closed contours with sufficient area
    contours, _ = cv2.findContours(
        canny_bnd.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    bright_bnd = np.zeros_like(canny_bnd, dtype=np.uint8)  # Changed to uint8
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            cv2.drawContours(bright_bnd, [contour], 0, 1, -1)
    
    return (pore_mask | (bright_bnd.astype(bool)) | pore_outline) & valid


def fft_gradient(gray, valid):
    h, w  = gray.shape
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    gx    = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad  = np.sqrt(gx**2 + gy**2).astype(np.float32)
    grad[~valid] = 0
    grad -= grad[valid].mean()

    H = int(2**np.ceil(np.log2(h)))
    W = int(2**np.ceil(np.log2(w)))
    pad         = np.zeros((H, W), np.float32)
    pad[:h, :w] = grad
    pad *= np.hanning(H).reshape(-1, 1) * np.hanning(W).reshape(1, -1)
    Fs   = np.fft.fftshift(np.fft.fft2(pad))
    PS_g = np.abs(Fs)**2

    cy, cx = H // 2, W // 2
    yi, xi = np.indices((H, W))
    r      = np.sqrt((yi - cy)**2 + (xi - cx)**2).astype(int)
    r_max  = min(H, W) // 2
    rpow   = np.bincount(r.ravel(), weights=PS_g.ravel(), minlength=r_max + 1)
    rcnt   = np.bincount(r.ravel(), minlength=r_max + 1).clip(1)
    rp     = rpow[:r_max] / rcnt[:r_max]
    freq   = np.arange(r_max) * PPU / W

    mask   = (freq >= 0.05) & (freq <= 1 / 0.8)
    fm, pm = freq[mask], rp[mask]
    n      = len(pm)
    win    = min(11, n if n % 2 == 1 else n - 1)
    win    = max(win, 3)
    pm_sm  = savgol_filter(pm, win, min(2, win - 1)) if n > 4 else pm
    pm_sm  = np.maximum(pm_sm, 0)
    peaks, props = find_peaks(pm_sm, prominence=pm_sm.max() * 0.01, distance=2)
    if len(peaks):
        best   = peaks[np.argmax(props['prominences'])]
        f_peak = fm[best]
    else:
        f_peak = fm[np.argmax(pm_sm)]

    d_fft  = 1.0 / f_peak
    r_pk_px= f_peak * W / PPU

    img_       = gray.astype(np.float32)
    img_[~valid] = img_[valid].mean()
    img_      -= img_[valid].mean()
    img_ *= np.hanning(h).reshape(-1, 1) * np.hanning(w).reshape(1, -1)
    pad2       = np.zeros((H, W), np.float32)
    pad2[:h,:w]= img_
    Fs2        = np.fft.fftshift(np.fft.fft2(pad2))
    r_all      = np.sqrt((yi - cy)**2 + (xi - cx)**2)
    bp = np.where((r_all >= r_pk_px * 0.4) & (r_all <= r_pk_px * 2.5), Fs2, 0+0j)
    bp_img     = np.fft.ifft2(np.fft.ifftshift(bp)).real[:h, :w]
    bp_img    -= bp_img.min()
    if bp_img.max() > 0:
        bp_img = bp_img / bp_img.max() * 255
    bp_img[~valid] = 0

    return d_fft, r_pk_px, bp_img, H, W, fm, pm_sm, f_peak


def greedy_pack_v4(dist, forbidden, r_guidance_px):
    grain_area = float((~forbidden).sum())
    if grain_area == 0:
        return []
    r_start = min(float(dist.max()), r_guidance_px * 2.0)
    r_stop  = max(1.5, r_guidance_px * 0.4)
    r_stop  = min(r_stop, r_start * 0.95)
    cx_a = np.empty(0, np.float32)
    cy_a = np.empty(0, np.float32)
    r_a  = np.empty(0, np.float32)
    placed = []
    stall  = 0
    r_cur  = r_start
    while r_cur >= r_stop:
        ys, xs = np.where(dist >= r_cur)
        if not len(ys):
            r_cur *= RADIUS_STEP
            continue
        proximity = dist[ys, xs]
        score     = proximity - BRIGHT_W * (proximity / (dist.max() + 1e-6))
        order     = np.argsort(score)
        ys, xs    = ys[order], xs[order]
        n_placed_this = 0
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            r = float(dist[cy, cx])
            r = min(r, r_cur * 1.05)
            if r < r_stop:
                continue
            if len(cx_a):
                d2      = (cx - cx_a)**2 + (cy - cy_a)**2
                min_sep = MIN_SEP_FRAC * (r + r_a)
                if np.any(d2 < min_sep**2):
                    continue
            placed.append((cx, cy, r))
            cx_a = np.append(cx_a, cx)
            cy_a = np.append(cy_a, cy)
            r_a  = np.append(r_a,  r)
            n_placed_this += 1
        gain = (np.pi * r_cur**2 * n_placed_this) / (grain_area + 1e-6)
        stall = stall + 1 if gain < COVER_THRESH else 0
        if stall >= 4:
            break
        r_cur *= RADIUS_STEP
    return placed


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def hr(char='─', n=62):
    print(char * n)


def main():
    hr('═')
    print('  sem_interactive_v2.py  |  演示模式  |  1100 °C  |  Panel f')
    hr('═')

    # Load image
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f'\n[ERROR] Cannot load image: {IMG_PATH}')
        sys.exit(1)

    crop  = img[PY:PY + PH, PX:PX + PW].copy()
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    valid = valid_mask(h, w)
    print(f'\n  Image loaded  |  Panel: {w}×{h} px  |  Output: {OUT}')

    # ────────────────────────────────────────────────────
    # Phase 1: Three pore separation methods → popup display
    # ────────────────────────────────────────────────────
    hr()
    print('  PHASE 1 — Pore Segmentation (3 methods)')
    hr()

    print('  Running Dark      (darkest 8% pixels)...')
    res_dark = segment_dark(gray, valid)

    print('  Running Bright    (brightest 8% pixels)...')
    res_bright = segment_bright(gray, valid)

    print('  Running Gradient  (highest gradient magnitude)...')
    res_grad = segment_gradient(gray, valid)

    results = {
        'Dark':       res_dark,
        'Bright':     res_bright,
        'Gradient':   res_grad,
    }

    print()
    print(f'  {"Method":<12} {"Pores":>6} {"Area(px)":>10} {"Area(%)":>8} {"k_size":>7}')
    print(f'  {"-"*12} {"-"*6} {"-"*10} {"-"*8} {"-"*7}')
    for name, (pm, seeds, k) in results.items():
        n = int(ndi.label(pm)[1])
        area = int(pm.sum())
        pct  = area / (h * w) * 100
        ks   = str(k) if k > 0 else 'N/A'
        print(f'  {name:<12} {n:>6} {area:>10} {pct:>7.1f}% {ks:>7}')

    print('\n  Displaying Phase 1... (close window to continue)')
    show_phase1(crop, results)

    # Wait for user selection
    print()
    print('  Select pore detection method:')
    print('    1 = Dark       (darkest 8% pixels)')
    print('    2 = Bright     (brightest 8% pixels)')
    print('    3 = Gradient   (highest gradient magnitude)')

    while True:
        choice = input('\n  Enter choice [1/2/3]: ').strip()
        if choice in ('1', '2', '3'):
            break
        print('  Please enter 1, 2, or 3.')

    method_map = {
        '1': ('Dark',      res_dark),
        '2': ('Bright',    res_bright),
        '3': ('Gradient',  res_grad),
    }
    sel_name, (pore_mask, pore_seeds, k_sel) = method_map[choice]
    print(f'\n  Selected: [{sel_name}]')

    # ────────────────────────────────────────────────────
    # Phase 2: Boundary + Distance Field (auto-executed)
    # ────────────────────────────────────────────────────
    hr()
    print('  PHASE 2 — Boundary + Distance Field (auto)')
    hr()

    bnd  = make_boundary(gray, valid, pore_mask)
    forb = (~valid) | bnd
    dist = ndi.distance_transform_edt((~forb).astype(np.uint8))

    print(f'  Boundary pixels : {int(bnd.sum())}')
    print(f'  Grain area      : {int((~forb).sum())} px ≈ {(~forb).sum() / PPU**2:.2f} µm²')
    print(f'  dist.max        : {dist.max():.1f} px = {dist.max() / PPU:.2f} µm')

    # ────────────────────────────────────────────────────
    # Phase 3: FFT Analysis
    # ────────────────────────────────────────────────────
    hr()
    print('  PHASE 3 — FFT Analysis')
    hr()

    d_fft, r_pk_px, bp_img, H, W, freq_arr, pow_sm, f_peak = fft_gradient(gray, valid)

    r_guidance = d_fft * CORRECTION / 2 * PPU
    if r_guidance < dist.max() * 0.35:
        r_guidance = dist.max()
        r_src = 'dist.max'
    else:
        r_src = 'FFT'

    print(f'  FFT peak freq   : {f_peak:.4f} µm⁻¹')
    print(f'  d_fft           : {d_fft:.3f} µm')
    print(f'  r_guidance      : {r_guidance / PPU:.3f} µm  [{r_src}]')

    # ────────────────────────────────────────────────────
    # Phase 4: Greedy Packing
    # ────────────────────────────────────────────────────
    hr()
    print('  PHASE 4 — Greedy Packing')
    hr()

    circs = greedy_pack_v4(dist, forb, r_guidance)

    if not circs:
        print('  [WARNING] No circles placed.')
        return

    d_um  = np.array([2 * r / PPU for _, _, r in circs])
    d50   = float(np.median(d_um))
    d_mean= float(d_um.mean())
    d_std = float(d_um.std())
    grain_area = float((~forb).sum())
    cov   = sum(np.pi * r**2 for _, _, r in circs) / (grain_area + 1e-6) * 100

    print(f'  Circles (N)  : {len(circs)}')
    print(f'  d50          : {d50:.3f} µm')
    print(f'  Mean         : {d_mean:.3f} µm')
    print(f'  Std          : {d_std:.3f} µm')
    print(f'  Coverage     : {cov:.1f}%')

    # ────────────────────────────────────────────────────
    # Merge final output: FFT bandpass (left) + Grain circles (right)
    # ────────────────────────────────────────────────────
    print()
    print('  Generating final combined result...')
    
    # FFT bandpass image (left)
    bp8 = bp_img.clip(0, 255).astype(np.uint8)
    bpc = cv2.applyColorMap(bp8, cv2.COLORMAP_VIRIDIS)
    bp_rgb = cv2.cvtColor(bpc, cv2.COLOR_BGR2RGB)
    
    # Grain circle image (right)
    ov_grain = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    ov_grain[pore_mask]            = ov_grain[pore_mask] * 0.25 + PORE_RGB * 0.75
    ov_grain[bnd & ~pore_mask]     = ov_grain[bnd & ~pore_mask] * 0.35 + BOUND_RGB * 0.65
    ov_grain = ov_grain.clip(0, 255).astype(np.uint8)
    
    # Draw grain circles
    rv = [rr for _, _, rr in circs]
    lo, hi = min(rv), max(rv)
    for cx, cy, rr in circs:
        norm = (rr - lo) / max(hi - lo, 1e-6)
        col  = [int(c * 255) for c in plt.cm.RdYlGn(1 - norm)[:3]]
        cv2.circle(ov_grain, (int(cx), int(cy)), max(1, int(rr)), col, 1, cv2.LINE_AA)
        cv2.circle(ov_grain, (int(cx), int(cy)), 2, (255, 255, 255), -1)
    
    # Create side-by-side image: FFT (left) + Grain circles (right)
    fig_final, axes_final = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_final.suptitle(
        f'Final Result  [{sel_name}]  |  1100 °C  |  '
        f'N={len(circs)}  d50={d50:.2f} µm  cov={cov:.1f}%',
        fontsize=13, fontweight='bold')
    
    # Left: FFT
    axes_final[0].imshow(bp_rgb)
    axes_final[0].set_title(f'FFT Bandpass  (d = {d_fft:.2f} µm)', fontsize=12, fontweight='bold')
    axes_final[0].axis('off')
    
    # Right: Grain circles
    axes_final[1].imshow(ov_grain)
    axes_final[1].set_title('Grain Circles (RdYlGn: large→small)', fontsize=12, fontweight='bold')
    axes_final[1].axis('off')
    
    plt.tight_layout()
    
    # Popup display
    print('  Displaying final result... (close window to save)')
    plt.show()
    
    # Save file
    out_final = str(OUT / 'final_result.png')
    fig_final.savefig(out_final, dpi=200, bbox_inches='tight')
    plt.close(fig_final)
    print(f'  → Saved: final_result.png')

    # ── Summary ───────────────────────────────────────────
    hr('═')
    print(f'  DONE  |  Method: {sel_name}  |  d50 = {d50:.3f} µm')
    hr('═')
    print(f'  Output file: {OUT / "final_result.png"}')
    hr('═')


if __name__ == '__main__':
    os.system('clear' if os.name != 'nt' else 'cls')
    main()
