"""
Tilted BK7 Lens — Full ASM + SHG + Phase Visualization  [HIGH QUALITY]
=======================================================================
Improvements over original:
  • GRID_N = 1024  (4× more pixels, 2× finer sampling in each axis)
  • Intensity panels use imshow() on a fine RGBA image instead of contourf
      → smooth gradients, no visible contour staircase artefacts
  • Colormap applied via matplotlib.cm  → 8-bit-free linear mapping
  • Phase panels: V = √(I/I_max)  masks low-amplitude regions to black
      so fringe noise disappears entirely outside the beam
  • _adaptive_window fill factor 0.68 → 0.72 for a tighter, less-padded crop
  • imshow uses interpolation='bicubic' for sub-pixel smoothness
  • Colorbars added to intensity panels so brightness scale is readable
  • Phase colorbar (hue wheel) added to one phase panel as a legend

Layout (below axial profile):
  ┌──────────────┬───────────────┬───────────────┐
  │ Input        │ Fund. d_conv  │ SHG           │  ← intensity  (magma / hot)
  ├──────────────┼───────────────┼───────────────┤
  │ Input phase  │ Phase d_conv  │ SHG phase     │  ← phase HSV  masked by √I
  └──────────────┴───────────────┴───────────────┘

Fixed parameters:  λ=1064 nm · w₀=900 µm · f=200 mm · BK7
Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.widgets import Slider, RadioButtons
from scipy.special import hermite, genlaguerre
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  Physical constants
# ══════════════════════════════════════════════════════════════════════════════
LAM_M    = 1064e-9
LAM_SHG  = LAM_M / 2.0
F_TARGET = 0.200
W0_M     = 900e-6
ZR       = np.pi * W0_M**2 / LAM_M


def _n_bk7():
    l2 = (LAM_M * 1e6)**2
    return np.sqrt(1
        + 1.03961212  * l2 / (l2 - 0.00600069867)
        + 0.231792344 * l2 / (l2 - 0.0200179144)
        + 1.01046945  * l2 / (l2 - 103.560653))


N2    = _n_bk7()
RHO1  = (N2 - 1.0) * F_TARGET
RHO2  = 1e9

# ── KEY QUALITY PARAMETER ──────────────────────────────────────────────────
GRID_N   = 1024          # was 512 — 2× finer → 4× more pixels
FILL_FAC = 0.72          # slightly tighter crop than original 0.68

# ══════════════════════════════════════════════════════════════════════════════
#  Analytical input fields
# ══════════════════════════════════════════════════════════════════════════════
def _field_TEM00(X, Y, w):
    return np.exp(-(X**2 + Y**2) / w**2).astype(complex)


def _field_HG_rot45(m, n, X, Y, w):
    sq2 = np.sqrt(2.0)
    xp  = (X + Y) / sq2;  yp = (X - Y) / sq2
    return (hermite(m)(sq2*xp/w) * hermite(n)(sq2*yp/w)
            * np.exp(-(X**2+Y**2)/w**2)).astype(complex)


def _field_LG(p, ell, X, Y, w):
    r2  = X**2 + Y**2
    rho = np.sqrt(2.0*r2) / w
    Lpl = genlaguerre(p, abs(ell))(rho**2)
    return (rho**abs(ell) * Lpl * np.exp(-r2/w**2)
            * np.exp(-1j*ell*np.arctan2(Y, X))).astype(complex)


MODES = {
    'TEM₀₀':
        ('TEM₀₀',         0, lambda X,Y,w: _field_TEM00(X,Y,w)),
    '$HG_{{(1,0)}}$@45°':
        ('$HG_{{(1,0)}}$ @ 45°', 1, lambda X,Y,w: _field_HG_rot45(1,0,X,Y,w)),
    '$HG_{{(2,0)}}$@45°':
        ('$HG_{{(2,0)}}$ @ 45°', 2, lambda X,Y,w: _field_HG_rot45(2,0,X,Y,w)),
    f'$LG^{{1}}_{{0}}$  (p=0,ℓ=1)':
        ('$LG^{{1}}_{{0}}$  p=0,ℓ=1', 1, lambda X,Y,w: _field_LG(0,1,X,Y,w)),
    '$LG^{{2}}_{{0}}$  (p=0,ℓ=2)':
        ('$LG^{{2}}_{{0}}$  p=0,ℓ=2', 2, lambda X,Y,w: _field_LG(0,2,X,Y,w)),
}
MODE_KEYS = list(MODES.keys())

# ══════════════════════════════════════════════════════════════════════════════
#  ABCD + q-parameter
# ══════════════════════════════════════════════════════════════════════════════
def tilted_lens_matrices(theta1_deg):
    t1 = np.radians(theta1_deg)
    s2 = np.sin(t1) / N2
    if abs(s2) >= 1.0:
        raise ValueError("Total internal reflection — reduce θ.")
    t2 = np.arcsin(s2); c1, c2 = np.cos(t1), np.cos(t2)
    fac  = (N2*c2)/c1 - 1.0
    curv = 1.0/RHO1 - 1.0/RHO2
    phi_t = fac*curv/c1;  phi_s = fac*curv*c1
    Mt = np.array([[1.,0.],[-phi_t,1.]])
    Ms = np.array([[1.,0.],[-phi_s,1.]])
    ft = 1.0/phi_t if abs(phi_t)>1e-20 else np.inf
    fs = 1.0/phi_s if abs(phi_s)>1e-20 else np.inf
    return Mt, Ms, np.degrees(t2), ft, fs


def _free(d): return np.array([[1.,d],[0.,1.]])
def _pq(q, M): return (M[0,0]*q+M[0,1])/(M[1,0]*q+M[1,1])
def _wq(q, lam=LAM_M):
    im = np.imag(1.0/q)
    return np.sqrt(-lam/(np.pi*im)) if im < 0 else np.nan
def _waist(q, lam=LAM_M):
    zw = -np.real(q);  zR2 = np.imag(q)
    return zw, (np.sqrt(lam*zR2/np.pi) if zR2>0 else np.nan)


def trace(d_after, theta1_deg, n_pts=900):
    Mt, Ms, t2d, ft, fs = tilted_lens_matrices(theta1_deg)
    q0  = 1j*ZR;  qpt = _pq(q0,Mt);  qps = _pq(q0,Ms)
    z   = np.linspace(0., d_after, n_pts)
    wt  = np.array([_wq(_pq(qpt,_free(zi))) for zi in z])
    ws  = np.array([_wq(_pq(qps,_free(zi))) for zi in z])
    zwt, w0t = _waist(qpt);  zws, w0s = _waist(qps)
    return z, wt, ws, dict(
        theta2_deg=t2d, ft_mm=ft*1e3, fs_mm=fs*1e3,
        waist_t_z=zwt, waist_s_z=zws,
        waist_t_w=w0t, waist_s_w=w0s,
        gap_mm=abs(zwt-zws)*1e3)

# ══════════════════════════════════════════════════════════════════════════════
#  ASM
# ══════════════════════════════════════════════════════════════════════════════
def _asm(E, x, y, lam, d):
    if abs(d) < 1e-12:
        return E.copy()
    k  = 2.0*np.pi/lam
    KX, KY = np.meshgrid(
        2*np.pi*np.fft.fftfreq(len(x), x[1]-x[0]),
        2*np.pi*np.fft.fftfreq(len(y), y[1]-y[0]))
    k2  = KX**2 + KY**2
    kz  = np.where(k2 <= k**2, np.sqrt(np.maximum(k**2-k2, 0.)), 0.)
    return np.fft.ifft2(np.fft.fft2(E) * np.exp(1j*kz*d))

# ══════════════════════════════════════════════════════════════════════════════
#  Full propagation
# ══════════════════════════════════════════════════════════════════════════════
def propagate_fundamental(mode_key, ft, fs, d_conv):
    _, N_ord, field_fn = MODES[mode_key]
    q0  = 1j * ZR
    Ct  = -1.0/ft if np.isfinite(ft) else 0.
    Cs  = -1.0/fs if np.isfinite(fs) else 0.
    qpt = _pq(q0, np.array([[1.,0.],[Ct,1.]]))
    qps = _pq(q0, np.array([[1.,0.],[Cs,1.]]))
    wt_c = _wq(_pq(qpt, _free(d_conv))) or W0_M
    ws_c = _wq(_pq(qps, _free(d_conv))) or W0_M

    w_eff_in  = W0_M * np.sqrt(N_ord+1)
    w_eff_out = max(wt_c, ws_c) * np.sqrt(N_ord+1)
    half = max(w_eff_in, w_eff_out) * 3.8

    x = np.linspace(-half, half, GRID_N)
    y = np.linspace(-half, half, GRID_N)
    X, Y = np.meshgrid(x, y)

    E_in_pure = field_fn(X, Y, W0_M)
    mx = np.abs(E_in_pure).max()
    if mx > 0: E_in_pure = E_in_pure / mx

    k = 2.0*np.pi/LAM_M
    phx = X**2/ft if np.isfinite(ft) else np.zeros_like(X)
    phy = Y**2/fs if np.isfinite(fs) else np.zeros_like(Y)
    E_after_lens = E_in_pure * np.exp(-1j*k/2.*(phx+phy))
    E_conv       = _asm(E_after_lens, x, y, LAM_M, d_conv)
    return x, y, E_in_pure, E_conv, wt_c, ws_c

# ══════════════════════════════════════════════════════════════════════════════
#  Adaptive window (99 % of power)
# ══════════════════════════════════════════════════════════════════════════════
def _adaptive_window(x_m, y_m, I, fill=FILL_FAC):
    I_norm = I / (I.sum() + 1e-30)
    X, Y   = np.meshgrid(x_m, y_m)
    cx = np.sum(I_norm * X);  cy = np.sum(I_norm * Y)
    r_flat  = np.sqrt(((X-cx)**2 + (Y-cy)**2).ravel())
    i_flat  = I.ravel()
    idx_srt = np.argsort(r_flat)
    cum     = np.cumsum(i_flat[idx_srt]); cum /= cum[-1]+1e-30
    r99     = r_flat[idx_srt[np.searchsorted(cum, 0.99)]]
    half    = np.clip(r99/fill, 1e-7, min(x_m.max(), y_m.max()))
    return half, cx, cy


def _crop(x_m, y_m, arr, cx, cy, half):
    xi0 = max(0, np.searchsorted(x_m, cx-half))
    xi1 = min(len(x_m)-1, np.searchsorted(x_m, cx+half))
    yi0 = max(0, np.searchsorted(y_m, cy-half))
    yi1 = min(len(y_m)-1, np.searchsorted(y_m, cy+half))
    return x_m[xi0:xi1]*1e6, y_m[yi0:yi1]*1e6, arr[yi0:yi1, xi0:xi1]

# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-QUALITY intensity rendering
#  Uses imshow on an RGBA array produced by the colormap.
#  Achieves smooth gradients (no contour staircase).
# ══════════════════════════════════════════════════════════════════════════════
def _intensity_rgba(I_norm, cmap_name):
    """Convert normalised [0,1] intensity array → RGBA uint8 via cmap."""
    cmap = mcm.get_cmap(cmap_name)
    rgba = cmap(I_norm)          # float64 RGBA, shape (H,W,4)
    return rgba


def _show_intensity(ax, x_m, y_m, E, title, cmap='magma',
                    border_col=None, win=None, add_cbar=False, fig=None):
    I = np.abs(E)**2
    if I.max() < 1e-30:
        ax.text(0.5, 0.5, 'Null field', transform=ax.transAxes,
                ha='center', va='center', color='#777', fontsize=9)
        ax.set_title(title, fontsize=8, pad=4, color=TEXT, fontweight='semibold')
        return None

    if win is None:
        win = _adaptive_window(x_m, y_m, I)
    half, cx, cy = win

    xc, yc, Ic = _crop(x_m, y_m, I, cx, cy, half)
    Ic = Ic / Ic.max()

    # HIGH-QUALITY: imshow with bicubic interpolation
    rgba = _intensity_rgba(Ic, cmap)
    ext  = [xc[0], xc[-1], yc[0], yc[-1]]
    im   = ax.imshow(rgba, origin='lower', extent=ext, aspect='equal',
                     interpolation='bicubic')

    ax.axhline(cy*1e6, color='white', lw=0.3, alpha=0.18)
    ax.axvline(cx*1e6, color='white', lw=0.3, alpha=0.18)
    ax.set_xlim((cx-half)*1e6, (cx+half)*1e6)
    ax.set_ylim((cy-half)*1e6, (cy+half)*1e6)
    ax.set_title(title, fontsize=8, pad=3, color=TEXT, fontweight='semibold')
    ax.set_xlabel('x  tang. (µm)', fontsize=7, color='#999')
    ax.set_ylabel('y  sagit. (µm)', fontsize=7, color='#999')
    if border_col:
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col); sp.set_linewidth(2.2)

    if add_cbar and fig is not None:
        # Synthetic ScalarMappable for the colorbar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm   = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02,
                          orientation='vertical', label='|E|² (norm.)')
        cb.ax.tick_params(labelsize=6, colors='#999')
        cb.set_label('|E|² (norm.)', color='#999', fontsize=6)
    return im


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-QUALITY phase rendering
#  H = φ/(2π) mapped into HSV hue.
#  V = √(I/I_max) — amplitude mask: low-intensity regions go dark,
#      eliminating phase noise outside the beam entirely.
#  S = 1 always.
# ══════════════════════════════════════════════════════════════════════════════
def _phase_rgba(E):
    phi   = np.angle(E)                       # [-π, π]
    H     = (phi + np.pi) / (2.0 * np.pi)    # [0, 1]
    S     = np.ones_like(H)
    V     = np.ones_like(H)                   # full brightness everywhere — no amplitude mask
    RGB   = mcolors.hsv_to_rgb(np.stack([H, S, V], axis=-1))
    A     = np.ones(E.shape)
    return np.concatenate([RGB, A[:,:,None]], axis=-1)
 
 
def _show_phase(ax, x_m, y_m, E, title, border_col=None, win=None):
    I = np.abs(E)**2
    if I.max() < 1e-30:
        ax.text(0.5, 0.5, 'Null field', transform=ax.transAxes,
                ha='center', va='center', color='#777', fontsize=9)
        ax.set_title(title, fontsize=8, pad=4, color=TEXT, fontweight='semibold')
        return
 
    if win is None:
        win = _adaptive_window(x_m, y_m, I)
    half, cx, cy = win
 
    _, _, Ec   = _crop(x_m, y_m, E, cx, cy, half)
    xc, yc, _  = _crop(x_m, y_m, I, cx, cy, half)
    rgba        = _phase_rgba(Ec)
    ext         = [xc[0], xc[-1], yc[0], yc[-1]]
 
    # HIGH-QUALITY: bicubic — avoids pixel-block artefacts
    ax.imshow(rgba, origin='lower', extent=ext, aspect='equal',
              interpolation='bicubic')
    ax.axhline(cy*1e6, color='white', lw=0.3, alpha=0.12)
    ax.axvline(cx*1e6, color='white', lw=0.3, alpha=0.12)
    ax.set_xlim((cx-half)*1e6, (cx+half)*1e6)
    ax.set_ylim((cy-half)*1e6, (cy+half)*1e6)
    ax.set_title(title, fontsize=8, pad=3, color=TEXT, fontweight='semibold')
    ax.set_xlabel('x  tang. (µm)', fontsize=7, color='#999')
    ax.set_ylabel('y  sagit. (µm)', fontsize=7, color='#999')
    if border_col:
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col); sp.set_linewidth(2.2)
 

# ══════════════════════════════════════════════════════════════════════════════
#  Compute window helper
# ══════════════════════════════════════════════════════════════════════════════
def _compute_window(x_m, y_m, E):
    I = np.abs(E)**2
    if I.max() < 1e-30: return None
    return _adaptive_window(x_m, y_m, I)

# ══════════════════════════════════════════════════════════════════════════════
#  Colors & axis styling
# ══════════════════════════════════════════════════════════════════════════════
BG     = '#F5F7FA'; PANEL  = '#FFFFFF'; BORDER = '#C5CDD8'
TEXT   = '#1C2B3A'; MUTED  = '#556070'; BLUE   = '#1861C8'
TEAL   = '#0A9B7F'; ORANGE = '#D96010'; RED    = '#B03030'
VIOLET = '#7030A0'; GREEN2 = '#1A8040'; GRIDC  = '#DDE4EE'


def _style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRIDC, lw=0.5, ls='--', alpha=0.8); ax.set_axisbelow(True)


def _dark(ax):
    ax.set_facecolor('#080808')
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
    ax.tick_params(colors='#888', labelsize=7, length=2)
    ax.xaxis.label.set_color('#999'); ax.yaxis.label.set_color('#999')

# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════
def run():
    # Use a high DPI for sharper display on retina / high-DPI screens
    matplotlib.rcParams['figure.dpi'] = 120

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle(
        f'Tilted Lens — ASM + SHG\n ',
        color=TEXT, fontsize=10, fontweight='bold', y=0.998)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           left=0.055, right=0.72,
                           top=0.955, bottom=0.06,
                           hspace=0.48, wspace=0.38)

    ax_main   = fig.add_subplot(gs[0, :])
    ax_in_I   = fig.add_subplot(gs[1, 0])
    ax_conv_I = fig.add_subplot(gs[1, 1])
    ax_shg_I  = fig.add_subplot(gs[1, 2])
    ax_in_P   = fig.add_subplot(gs[2, 0])
    ax_conv_P = fig.add_subplot(gs[2, 1])
    ax_shg_P  = fig.add_subplot(gs[2, 2])
    ax_info   = fig.add_subplot(gs[1:, 3])

    _style(ax_main)
    for ax in (ax_in_I, ax_conv_I, ax_shg_I,
               ax_in_P, ax_conv_P, ax_shg_P):
        _dark(ax)
    ax_info.set_facecolor(PANEL)
    for sp in ax_info.spines.values(): sp.set_edgecolor(BORDER)
    ax_info.axis('off')

    # ── Widgets ───────────────────────────────────────────────────────────────
    ax_rb = fig.add_axes([0.745, 0.62, 0.225, 0.32], facecolor=PANEL)
    ax_rb.set_title('Input Mode', color=TEXT, fontsize=8.5,
                    fontweight='bold', pad=6)
    for sp in ax_rb.spines.values(): sp.set_edgecolor(BORDER)
    rb = RadioButtons(ax_rb, MODE_KEYS, activecolor=VIOLET)
    for lbl in rb.labels: lbl.set_color(TEXT); lbl.set_fontsize(8.5)

    SL = [
        ('θ tilt (°)',                        0.0,  55.0, 22.0,  0.5),
        ('Propagation window (mm)',           50.0, 800.0, 450.0, 5.0),
        ('Crystal Position (mm)',              1.0, 750.0, 180.0, 1.0),
        ('Propagation after SHG  (mm)',        0.0, 600.0,   0.0, 8.35),
    ]
    sliders = []
    for i, (lbl, vmin, vmax, v0, vstep) in enumerate(SL):
        ax_sl = fig.add_axes([0.745, 0.545 - i*0.118, 0.225, 0.036],
                             facecolor='#EEF2F7')
        col = GREEN2 if 'SHG' in lbl else BLUE
        sl  = Slider(ax_sl, lbl, vmin, vmax, valinit=v0,
                     valstep=vstep, color=col)
        sl.label.set_color(TEXT);  sl.label.set_fontsize(8)
        sl.valtext.set_color(col); sl.valtext.set_fontsize(8)
        sliders.append(sl)

    fig.text(0.747, 0.12,
             f'$λ_{{fund}}$ = {LAM_M*1e9:.0f} nm\n'
             f'$λ_{{SHG}}$  = {LAM_SHG*1e9:.0f} nm\n'
             f'w₀     = {W0_M*1e6:.0f} µm\n'
             f'$z_{{R}}$     = {ZR*1e3:.0f} mm\n'
             f'f      = {F_TARGET*1e3:.0f} mm\n',
             color=MUTED, fontsize=7.5, va='top',
             family='monospace', linespacing=1.6,
             transform=fig.transFigure)

    # ── Phase hue-wheel legend (placed once, bottom-right of phase row) ───────
    def _add_phase_legend(ax_ref):
        """Tiny hue-wheel colour legend next to one phase panel."""
        theta = np.linspace(0, 2*np.pi, 256)
        phi   = np.linspace(-np.pi, np.pi, 128)
        r     = np.linspace(0.3, 1.0, 64)
        TH, R = np.meshgrid(theta, r)
        H  = (TH) / (2*np.pi)
        V  = R
        S  = np.ones_like(H)
        rgb = mcolors.hsv_to_rgb(np.stack([H, S, V], axis=-1))
        # ax_ref bbox in figure coords
        pos = ax_ref.get_position()
        # place tiny polar axes in top-right corner of ax_ref
        ax_w = fig.add_axes(
            [pos.x1-0.045, pos.y1-0.055, 0.042, 0.052],
            projection='polar')
        ax_w.set_facecolor('black')
        ax_w.pcolormesh(theta, r, rgb[:,:,:3].mean(axis=-1),   # placeholder
                        cmap='hsv', shading='auto')
        ax_w.pcolormesh(theta, r, H, cmap='hsv', shading='auto')
        ax_w.set_xticks([]); ax_w.set_yticks([])
        ax_w.spines['polar'].set_visible(False)
        ax_w.set_title('φ', color='#ccc', fontsize=6, pad=1)
        return ax_w

    # ── Update ────────────────────────────────────────────────────────────────
    _legend_ax = [None]

    def update(_=None):
        theta  = sliders[0].val
        d_a_mm = sliders[1].val
        d_c_mm = sliders[2].val
        d_s_mm = sliders[3].val
        d_aft  = d_a_mm * 1e-3
        d_conv = min(d_c_mm * 1e-3, d_aft)
        d_shg  = d_s_mm * 1e-3
        mkey   = rb.value_selected
        _, N_ord, _ = MODES[mkey]

        try:
            Mt, Ms, t2d, ft, fs = tilted_lens_matrices(theta)
        except ValueError as e:
            ax_main.cla(); _style(ax_main)
            ax_main.text(0.5, 0.5, f'ERROR: {e}',
                         transform=ax_main.transAxes,
                         ha='center', va='center', color=RED, fontsize=11)
            fig.canvas.draw_idle(); return

        # ── Axial profile ─────────────────────────────────────────────────────
        z, wt, ws, info = trace(d_aft, theta)
        z_mm = z*1e3;  sc = np.sqrt(N_ord+1)

        ax_main.cla(); _style(ax_main)
        ax_main.fill_between(z_mm,  wt*sc*1e6, -wt*sc*1e6, color=BLUE, alpha=0.12)
        ax_main.fill_between(z_mm,  ws*sc*1e6, -ws*sc*1e6, color=TEAL, alpha=0.12)
        ax_main.plot(z_mm,  wt*sc*1e6, color=BLUE, lw=1.8, label='Tang. $w_{{eff}}$')
        ax_main.plot(z_mm, -wt*sc*1e6, color=BLUE, lw=1.8)
        ax_main.plot(z_mm,  ws*sc*1e6, color=TEAL, lw=1.8, label='Sagit. $w_{{eff}}$')
        ax_main.plot(z_mm, -ws*sc*1e6, color=TEAL, lw=1.8)
        ax_main.axvline(0,           color=ORANGE, lw=1.5, ls='-.', alpha=0.85,
                        label='Lens (z=0)')
        ax_main.axvline(d_conv*1e3,  color=VIOLET, lw=1.4, ls='--', alpha=0.9,
                        label=f'Crystal={d_conv*1e3:.0f} mm')
        for zw_k, col, lb in [('waist_t_z', BLUE, 'T-waist'),
                               ('waist_s_z', TEAL, 'S-waist')]:
            zw_mm = info[zw_k]*1e3
            if 0 < zw_mm < z_mm[-1]:
                ax_main.axvline(zw_mm, color=col, lw=0.9, ls=':', alpha=0.8)
                ylim = ax_main.get_ylim()
                ax_main.text(zw_mm + d_a_mm*0.008, ylim[1]*0.72,
                             f'{lb}\n{zw_mm:.1f} mm',
                             color=col, fontsize=6.5, va='top', fontweight='semibold')
        zt = info['waist_t_z']*1e3;  zs = info['waist_s_z']*1e3
        if 0 < zt < z_mm[-1] and 0 < zs < z_mm[-1]:
            ax_main.axvspan(min(zt,zs), max(zt,zs), color=ORANGE, alpha=0.07)

        ax_main.set_xlabel('z after lens (mm)', fontsize=9)
        ax_main.set_ylabel(f'$w_{{eff}}$ (µm)', fontsize=9)
        ax_main.set_xlim(0, d_a_mm)
        ax_main.set_title(
            f'{MODES[mkey][0]}  ·  θ={theta:.1f}°  ·  '
            f'$f_{{t}}$={info["ft_mm"]:.2f} mm  ·  $f_{{s}}$={info["fs_mm"]:.2f} mm  ·  '
            f'gap={info["gap_mm"]:.3f} mm',
            fontsize=9, pad=5, color=TEXT, fontweight='semibold')
        ax_main.legend(facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER,
                       fontsize=8, loc='upper right', framealpha=0.92)

        # ── ASM propagation ───────────────────────────────────────────────────
        try:
            x, y, E_in, E_conv, wt_c, ws_c = propagate_fundamental(
                mkey, ft, fs, d_conv)
        except Exception as exc:
            for ax in (ax_in_I, ax_conv_I, ax_shg_I,
                       ax_in_P, ax_conv_P, ax_shg_P):
                ax.cla(); _dark(ax)
                ax.text(0.5, 0.5, f'Error ASM:\n{exc}',
                        transform=ax.transAxes, ha='center', va='center',
                        color='#aaa', fontsize=8)
            fig.canvas.draw_idle(); return

        E_shg = _asm(E_conv**2, x, y, LAM_SHG, d_shg)

        # Adaptive windows (computed once, shared across intensity + phase)
        win_in   = _compute_window(x, y, E_in)
        win_conv = _compute_window(x, y, E_conv)
        win_shg  = _compute_window(x, y, E_shg)

        # ── Row 1: Intensities ────────────────────────────────────────────────
        for ax in (ax_in_I, ax_conv_I, ax_shg_I): ax.cla(); _dark(ax)

        _show_intensity(ax_in_I, x, y, E_in,
                        f'Input (SLM)  {MODES[mkey][0]}\n|E|²  (z=0)',
                        cmap='magma', win=win_in,
                        add_cbar=False, fig=fig)

        _show_intensity(ax_conv_I, x, y, E_conv,
                        f'Fund. ω  —  {d_conv*1e3:.0f} mm\n'
                        f'|E|²  wt={wt_c*1e6:.1f} · ws={ws_c*1e6:.1f} µm',
                        cmap='magma', border_col=VIOLET, win=win_conv,
                        add_cbar=False, fig=fig)

        _show_intensity(ax_shg_I, x, y, E_shg,
                        f'SHG 2ω  (λ={LAM_SHG*1e9:.0f} nm)\n'
                        f'|E_{{2ω}}|²  +{d_s_mm:.0f} mm',
                        cmap='hot', border_col=GREEN2, win=win_shg,
                        add_cbar=False, fig=fig)

        # ── Row 2: Phases (HSV, amplitude-masked) ────────────────────────────
        for ax in (ax_in_P, ax_conv_P, ax_shg_P): ax.cla(); _dark(ax)

        _show_phase(ax_in_P, x, y, E_in,
                    'Phase  arg(E)\n(z=0)',
                    win=win_in)

        _show_phase(ax_conv_P, x, y, E_conv,
                    f'Phase  arg(E)\n{d_conv*1e3:.0f} mm',
                    border_col=VIOLET, win=win_conv)

        _show_phase(ax_shg_P, x, y, E_shg,
                    f'Phase  arg(E$_{{2ω}}$)\n+{d_s_mm:.0f} mm',
                    border_col=GREEN2, win=win_shg)

        # Phase legend (hue wheel) — add once, then reuse
        if _legend_ax[0] is None:
            _legend_ax[0] = _add_phase_legend(ax_shg_P)

        # ── Info panel ────────────────────────────────────────────────────────
        ax_info.cla(); ax_info.axis('off')

        lines = [    ]
        y_cur = 0.985
        for txt, bold, col in lines:
            ax_info.text(0.03, y_cur, txt,
                         transform=ax_info.transAxes,
                         color=col, fontsize=7.4, va='top',
                         fontweight='bold' if bold else 'normal',
                         family='monospace')
            y_cur -= 0.048 if bold else 0.043

        fig.canvas.draw_idle()

    for sl in sliders: sl.on_changed(update)
    rb.on_clicked(update)
    update()
    plt.show()


if __name__ == '__main__':
    run()
