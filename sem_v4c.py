"""
SEM grain analysis v4
Changes vs v3:
  Pore:  dark p8 seeds → morph_close(adaptive k) → flood-fill → closed pore mask
  Pack:  no n_target; FFT r_target = guidance only (R_MIN=0.4, R_MAX=2.0×r_target)
         circles may overlap up to 10% (OVERLAP_TOL=0.90→min_sep=0.90*(r1+r2))
         packing stops when marginal coverage gain < threshold (geometric feedback)
         radius stepping: r_max→r_min, step×0.85
"""

import cv2, numpy as np, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from scipy.signal import find_peaks, savgol_filter

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_PATH = SCRIPT_DIR / 'SEM_RAW.png'
OUT = SCRIPT_DIR / 'outputs'
OUT.mkdir(exist_ok=True)

PPU      = 180 / 10.0   # 18.0 px/µm
BORDER   = 3

PANELS = [
    ('c', 515,  64, 514, 255, 1050, 'gradient'),
    ('e',   9, 412, 510, 255, 1090, 'gradient'),
    ('f', 519, 412, 510, 255, 1100, 'gradient'),
]

CORRECTION    = 0.50
MIN_SEP_FRAC  = 0.90    # circles overlap allowed: centres >= 0.90*(r1+r2)
RADIUS_STEP   = 0.85    # shrink factor per radius level
COVER_THRESH  = 0.002   # stop if coverage gain < 0.2% per 30 new circles
BRIGHT_W      = 0.3

img = cv2.imread(IMG_PATH)

# ═══════════════════════════════════════════════
# Step 1: Pore segmentation (seed → grow → fill)
# ═══════════════════════════════════════════════

def segment_pores(gray, valid):
    """
    A: dark p8 → morph_open (denoise) → connected-component seeds
       Discard any seed CC that does NOT form a closed enclosure
       (closed = the CC, when dilated 1px, forms a loop with no interior hole)
       Simplified criterion: keep only CCs whose convex-hull area > 10px
    B: adaptive morph_close → flood-fill each surviving seed
       Remove filled regions with area < 10px (too small to be a real pore)
    Returns: pore_mask (bool), seeds_kept (bool), k_size (int)
    """
    px   = gray[valid]
    p8   = np.percentile(px, 8)
    h, w = gray.shape

    # ── Step A: seed detection ──────────────────────────────────────────
    raw_dark = (gray <= p8) & valid
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seeds_raw = cv2.morphologyEx(
        raw_dark.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)

    # Label every connected component
    n_cc, cc_labels = cv2.connectedComponents(seeds_raw.astype(np.uint8))

    # Keep only CCs that can potentially enclose area:
    #   criterion: pixel count >= 6  (need at least 6 px to form a loop)
    #   AND the CC is not a single straight line
    #   Simple proxy: keep if area >= 6 px
    MIN_SEED_PX = 6
    kept_seeds = np.zeros((h, w), bool)
    for lbl in range(1, n_cc):
        mask_cc = (cc_labels == lbl)
        area = int(mask_cc.sum())
        if area >= MIN_SEED_PX:
            kept_seeds |= mask_cc

    # ── Adaptive kernel from kept seeds ────────────────────────────────
    dist_seed = ndi.distance_transform_edt(~kept_seeds)
    seed_vals = dist_seed[kept_seeds]
    if len(seed_vals) > 0:
        median_r = float(np.median(seed_vals[seed_vals < 50]))
    else:
        median_r = 3.0
    k_size = max(5, int(median_r * 3) | 1)

    # ── Step B: grow → fill → filter by enclosed area ──────────────────
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    pore_closed = cv2.morphologyEx(
        kept_seeds.astype(np.uint8), cv2.MORPH_CLOSE, k_grow)

    # Flood-fill to make solid
    pore_filled = ndi.binary_fill_holes(pore_closed.astype(bool))

    # Label filled regions and remove those with area < 10px
    MIN_PORE_AREA = 10
    n_filled, filled_labels = cv2.connectedComponents(
        pore_filled.astype(np.uint8))
    pore_mask = np.zeros((h, w), bool)
    for lbl in range(1, n_filled):
        region = (filled_labels == lbl)
        if int(region.sum()) >= MIN_PORE_AREA:
            pore_mask |= region

    pore_mask &= valid

    return pore_mask, kept_seeds, k_size

# ═══════════════════════════════════════════════
# Step 2: Auto boundary (pore outline + bright edges)
# ═══════════════════════════════════════════════

def make_boundary(gray, valid, pore_mask):
    """
    Boundary = pore outline (morphological gradient of pore_mask)
             + Canny of top-3% bright pixels (grain boundary highlights)
    """
    px  = gray[valid]
    k3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k1  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

    # Pore outline: dilate(pore) XOR pore
    pore_u8  = pore_mask.astype(np.uint8)
    dilated  = cv2.dilate(pore_u8, k3)
    pore_outline = (dilated - pore_u8).astype(bool) & valid

    # Bright Canny
    bt    = np.percentile(px, 97)
    canny = cv2.Canny((gray >= bt).astype(np.uint8)*255, 50, 150)
    bright_bnd = (canny > 0) & valid

    bnd = (pore_mask | pore_outline | bright_bnd) & valid
    return bnd

# ═══════════════════════════════════════════════
# Step 3: FFT (gradient magnitude, grain spacing)
# ═══════════════════════════════════════════════

def fft_gradient(gray, valid, ppu):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gx   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2+gy**2).astype(np.float32)
    grad[~valid] = 0; grad -= grad[valid].mean()

    H = int(2**np.ceil(np.log2(h)))
    W = int(2**np.ceil(np.log2(w)))
    pad = np.zeros((H,W), np.float32); pad[:h,:w] = grad
    pad *= np.hanning(H).reshape(-1,1) * np.hanning(W).reshape(1,-1)
    Fs   = np.fft.fftshift(np.fft.fft2(pad))
    PS_g = np.abs(Fs)**2

    cy, cx = H//2, W//2
    yi, xi = np.indices((H,W))
    r      = np.sqrt((yi-cy)**2+(xi-cx)**2).astype(int)
    r_max  = min(H,W)//2
    rpow   = np.bincount(r.ravel(), weights=PS_g.ravel(), minlength=r_max+1)
    rcnt   = np.bincount(r.ravel(), minlength=r_max+1).clip(1)
    rp     = rpow[:r_max] / rcnt[:r_max]
    freq   = np.arange(r_max) * ppu / W

    mask   = (freq >= 0.05) & (freq <= 1/0.8)
    fm, pm = freq[mask], rp[mask]
    n = len(pm); win = min(11, n if n%2==1 else n-1); win = max(win, 3)
    pm_sm  = savgol_filter(pm, win, min(2,win-1)) if n>4 else pm
    pm_sm  = np.maximum(pm_sm, 0)
    peaks, props = find_peaks(pm_sm, prominence=pm_sm.max()*0.01, distance=2)

    if len(peaks):
        best   = peaks[np.argmax(props['prominences'])]
        f_peak = fm[best]
    else:
        f_peak = fm[np.argmax(pm_sm)]

    d_fft    = 1.0 / f_peak
    r_pk_px  = f_peak * W / ppu

    # Bandpass on gray
    img_ = gray.astype(np.float32)
    img_[~valid] = img_[valid].mean(); img_ -= img_[valid].mean()
    img_ *= np.hanning(h).reshape(-1,1) * np.hanning(w).reshape(1,-1)
    pad2 = np.zeros((H,W),np.float32); pad2[:h,:w] = img_
    Fs2  = np.fft.fftshift(np.fft.fft2(pad2))
    r_all= np.sqrt((yi-cy)**2+(xi-cx)**2)
    bp   = np.where((r_all>=r_pk_px*0.4)&(r_all<=r_pk_px*2.5), Fs2, 0+0j)
    bp_img = np.fft.ifft2(np.fft.ifftshift(bp)).real[:h,:w]
    bp_img -= bp_img.min()
    if bp_img.max()>0: bp_img = bp_img/bp_img.max()*255
    bp_img[~valid] = 0

    return d_fft, r_pk_px, bp_img, H, W

# ═══════════════════════════════════════════════
# Step 4: Radius-stepping greedy pack
#   - No n_target: pack as many as geometry allows
#   - FFT r_target is guidance: R_MIN=0.4×, R_MAX=2.0×
#   - Overlap: centres >= MIN_SEP_FRAC*(r1+r2)  (10% overlap allowed)
#   - Stopping: geometric feedback (coverage gain < COVER_THRESH)
# ═══════════════════════════════════════════════

def greedy_pack_v4(dist, forbidden, r_guidance_px):
    """
    Radius-stepping packing:
      r_start = min(dist.max(), r_guidance_px * 2.0)
      r_stop  = max(1.5px, r_guidance_px * 0.4)
      step    = × RADIUS_STEP each level
    No n_target; stops when coverage gain saturates.
    """
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
    covered = 0.0   # approximate covered area (sum of π r²)
    stall   = 0

    r_cur = r_start
    while r_cur >= r_stop:
        ys, xs = np.where(dist >= r_cur)
        if not len(ys):
            r_cur *= RADIUS_STEP; continue

        # Sort: boundary-tangent first (smallest dist = closest to pore/boundary)
        proximity = dist[ys, xs]
        score = proximity - BRIGHT_W * (proximity / (dist.max()+1e-6))
        order = np.argsort(score)          # ascending: boundary-tangent first
        ys, xs = ys[order], xs[order]

        n_placed_this_level = 0
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            r = float(dist[cy, cx])
            r = min(r, r_cur * 1.05)      # allow slight size variation
            if r < r_stop: continue

            if len(cx_a):
                d2 = (cx-cx_a)**2 + (cy-cy_a)**2
                min_sep = MIN_SEP_FRAC * (r + r_a)
                if np.any(d2 < min_sep**2):
                    continue

            placed.append((cx, cy, r))
            cx_a = np.append(cx_a, cx)
            cy_a = np.append(cy_a, cy)
            r_a  = np.append(r_a,  r)
            covered += np.pi * r**2
            n_placed_this_level += 1

        # Coverage feedback: if gain this level tiny, start counting stalls
        gain = (np.pi * r_cur**2 * n_placed_this_level) / (grain_area + 1e-6)
        if gain < COVER_THRESH:
            stall += 1
        else:
            stall = 0
        if stall >= 4:     # 4 consecutive low-gain levels → stop
            break

        r_cur *= RADIUS_STEP

    return placed

# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def vm(h, w):
    m = np.ones((h,w), bool)
    m[:BORDER,:]=False; m[-BORDER:,:]=False
    m[:,:BORDER]=False; m[:,-BORDER:]=False
    return m

BOUND_RGB = np.array([45, 82, 160], np.float32)
PORE_RGB  = np.array([180, 60, 200], np.float32)   # purple = pore fill
results   = {}

print(f"{'Tag':>4}  {'T':>6}  {'k_pore':>7}  {'d_fft':>7}  "
      f"{'r_guid':>7}  {'N':>5}  {'d50':>7}  {'mean':>7}  {'cov%':>6}")
print("-"*78)

for tag,x,y,w,h,temp,fft_mode in PANELS:
    crop  = img[y:y+h, x:x+w].copy()
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hh,ww = gray.shape
    v     = vm(hh, ww)

    # Pore segmentation
    pore_mask, pore_seeds, k_size = segment_pores(gray, v)

    # Boundary from pore outline + bright edges
    bnd  = make_boundary(gray, v, pore_mask)
    forb = (~v) | bnd
    dist = ndi.distance_transform_edt((~forb).astype(np.uint8))

    # FFT guidance
    d_fft, r_pk, bp_img, H, W = fft_gradient(gray, v, PPU)

    # r_guidance: FFT primary, fallback to dist.max
    r_guidance = d_fft * CORRECTION / 2 * PPU
    if r_guidance < dist.max() * 0.35:
        r_guidance = dist.max()
        r_src = 'dist.max'
    else:
        r_src = 'FFT'

    # Greedy packing v4
    circs = greedy_pack_v4(dist, forb, r_guidance)
    d_um  = np.array([2*r/PPU for _,_,r in circs])
    d50   = float(np.median(d_um)) if len(d_um) else 0
    mean  = float(d_um.mean())     if len(d_um) else 0
    grain_area = float((~forb).sum())
    cov   = sum(np.pi*r**2 for _,_,r in circs) / (grain_area+1e-6) * 100

    results[tag] = dict(
        crop=crop, bnd=bnd, pore_mask=pore_mask, pore_seeds=pore_seeds,
        circs=circs, d_um=d_um, bp_img=bp_img,
        d_fft=d_fft, r_guidance=r_guidance, r_src=r_src,
        H=H, W=W, temp=temp, k_size=k_size, cov=cov)

    print(f"  {tag:>2}  {temp:>6}  {k_size:>7}  {d_fft:>7.3f}  "
          f"{r_guidance/PPU:>7.3f}  {len(circs):>5}  "
          f"{d50:>7.3f}  {mean:>7.3f}  {cov:>6.1f}%  [{r_src}]")

# ── Figure ─────────────────────────────────────────────────────────────
import matplotlib as mpl
mpl.rc('font', family='Liberation Sans', size=14)

fig, axes = plt.subplots(3, 2, figsize=(11, 12))
fig.subplots_adjust(left=0.08, right=0.98, top=0.97,
                    bottom=0.04, hspace=0.06, wspace=0.04)

col_labels = ['FFT Bandpass', 'SEM Circles']
for j, lbl in enumerate(col_labels):
    axes[0,j].set_title(lbl, fontsize=14, fontfamily='Liberation Sans',
                         fontweight='bold', pad=4)

panel_temps = {'c': 1050, 'e': 1090, 'f': 1100}

for ri, tag in enumerate(['c','e','f']):
    r   = results[tag]
    d   = r['d_um']
    d50 = float(np.median(d)) if len(d) else 0
    mn  = float(d.mean())     if len(d) else 0

    # Left: bandpass
    bp  = r['bp_img'].clip(0,255).astype(np.uint8)
    bpc = cv2.applyColorMap(bp, cv2.COLORMAP_VIRIDIS)
    bpr = cv2.cvtColor(bpc, cv2.COLOR_BGR2RGB)
    axes[ri,0].imshow(bpr)
    axes[ri,0].set_ylabel(f"{r['temp']} °C",
                           fontsize=14, fontfamily='Liberation Sans',
                           fontweight='bold', labelpad=4)
    axes[ri,0].set_xlabel(
        f"d = {r['d_fft']:.2f} µm",
        fontsize=14, fontfamily='Liberation Sans', labelpad=3)
    axes[ri,0].tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
    for sp in axes[ri,0].spines.values():
        sp.set_linewidth(0.5)

    # Right: SEM + pore + circles
    ov = cv2.cvtColor(r['crop'], cv2.COLOR_BGR2RGB).astype(np.float32)
    ov[r['pore_mask']] = ov[r['pore_mask']]*0.25 + PORE_RGB*0.75
    ov[r['bnd'] & ~r['pore_mask']] = (
        ov[r['bnd'] & ~r['pore_mask']]*0.35 + BOUND_RGB*0.65)
    ov = ov.clip(0,255).astype(np.uint8)

    rv  = [rr for _,_,rr in r['circs']]
    lo, hi = (min(rv),max(rv)) if rv else (1,2)
    for cx,cy,rr in r['circs']:
        norm = (rr-lo)/max(hi-lo,1e-6)
        col  = [int(c*255) for c in plt.cm.RdYlGn(1-norm)[:3]]
        cv2.circle(ov,(int(cx),int(cy)),max(1,int(rr)),col,1,cv2.LINE_AA)
        cv2.circle(ov,(int(cx),int(cy)),2,(255,255,255),-1)

    axes[ri,1].imshow(ov)
    axes[ri,1].set_xlabel(
        f"N = {len(d)}    d50 = {d50:.2f} µm    mean = {mn:.2f} µm",
        fontsize=14, fontfamily='Liberation Sans', labelpad=3)
    axes[ri,1].tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
    for sp in axes[ri,1].spines.values():
        sp.set_linewidth(0.5)

print('\nDisplaying result... (close window to save)')
plt.show()  # ← 加这一行

fig.savefig(str(OUT/'sem_v4_result.jpg'), dpi=350, bbox_inches='tight')
print('\nSaved: sem_result.jpg')
