#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNSS / GPS L1 C/A Software Receiver & Spoofing Attack Simulator  (v2 - fixed)
=============================================================================
Single-file PyQt5 + PyQtGraph + SciPy application.

  * Synthetic complex-baseband GPS L1 C/A signal: 3 PRNs, configurable Doppler and
    code delay, 50 bps nav data, AWGN at a configurable C/N0 (fs = 2.046 MHz)
  * Acquisition: FFT-based parallel code-phase search over Doppler bins, plus
    parabolic Doppler and code refinement
  * Tracking: 2nd-order DLL (normalized early-minus-late envelope, d = 0.5 chip)
              3rd-order Costas PLL with continuous 2nd-order FLL assist
  * Spoofer: a code- and carrier-consistent counterfeit of one PRN. It starts phase-aligned
    with the true signal and ramps its power above it. It then applies a smooth
    (minimum-jerk), accelerating code-delay offset that "carries off" the DLL and PLL.
  * Live 3D correlation surface (code delay x Doppler), DLL/PLL discriminator plots,
    tracking error vs. truth, and a lock-status table

v2 fixes:
  * 3D surface crash "IndexError: index 61 is out of bounds for axis 0 with size 61"
    (vertex colours are now passed to MeshData as a flat (nx*ny, 4) array)
  * Smoother carry-off profile + continuous FLL assist so the loops follow the
    spoofer cleanly instead of dropping into loss-of-lock half-way
  * Robust GL setup on older pyqtgraph versions

Requirements:  pip install numpy scipy pyqt5 pyqtgraph PyOpenGL
"""
import sys
import math
from collections import deque

import numpy as np
from scipy import fft as sfft
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
    HAVE_GL = True
except Exception:  # PyOpenGL missing -> 2D heat-map fallback
    HAVE_GL = False

# ============================== CONSTANTS ==============================
F_L1 = 1575.42e6
F_CHIP = 1.023e6
CODE_LEN = 1023
FS = 2.046e6                       # complex baseband, 2 samples/chip
N_SAMP = int(round(FS * 1e-3))     # 1 ms block = one C/A code period
T_INT = 1e-3
CARR_PER_CHIP = F_L1 / F_CHIP      # 1540 carrier cycles per chip
CHIP_M = 299792458.0 / F_CHIP      # ~293 m per chip
T_VEC = np.arange(N_SAMP) / FS
EL_SPACING = 0.5                   # early/late offset from prompt [chips]

# Loop parameters (Kaplan & Hegarty loop-filter forms)
PLL_BN, FLL_BN, DLL_BN = 18.0, 8.0, 2.0
W0P = PLL_BN / 0.7845              # 3rd-order PLL
A3, B3 = 1.1, 2.4
W0F = FLL_BN / 0.53                # 2nd-order FLL assist (always on)
A2 = 1.414
W0D = DLL_BN / 0.53                # 2nd-order DLL

ACQ_DOPP = np.arange(-5000.0, 5000.1, 100.0)    # acquisition Doppler bins [Hz]
VIEW_DOPP = np.arange(-2500.0, 2500.1, 125.0)   # live 3D view Doppler offsets [Hz]
VIEW_CODE = np.arange(-30, 31)                  # live 3D view code offsets [samples] (±15 chips)
VIEW_WIPE = np.exp(-2j * np.pi * VIEW_DOPP[:, None] * T_VEC[None, :])
ACQ_WIPE = np.exp(-2j * np.pi * ACQ_DOPP[:, None] * T_VEC[None, :])
HIST = 6000                                     # 6 s of 1 ms epochs in the plots

DEFAULT_SATS = [(3, 1250.0, 120.30), (11, -2340.0, 431.60), (19, 3410.0, 802.10)]
CH_COLORS = ['#ff5c5c', '#4cd964', '#4da3ff']

# ============================== C/A CODES ==============================
G2_TAPS = {1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10), 7: (1, 8),
           8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8),
           15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7), 21: (5, 8),
           22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10),
           29: (1, 6), 30: (2, 7), 31: (3, 8), 32: (4, 9)}


def ca_code(prn):
    """Generate the 1023-chip GPS C/A Gold code (+1/-1) for a PRN using the G1/G2 LFSRs."""
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    t1, t2 = G2_TAPS[prn]
    out = np.empty(CODE_LEN, dtype=np.int8)
    for i in range(CODE_LEN):
        out[i] = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
        f1 = g1[2] ^ g1[9]                                    # G1 = 1 + x^3 + x^10
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]    # G2 = 1+x^2+x^3+x^6+x^8+x^9+x^10
        g1 = np.roll(g1, 1); g1[0] = f1
        g2 = np.roll(g2, 1); g2[0] = f2
    return (1 - 2 * out.astype(np.float64))


CA = {p: ca_code(p) for p in range(1, 33)}
_SAMPLE_CHIP = (np.arange(N_SAMP) * F_CHIP / FS).astype(np.int64) % CODE_LEN
CODE_FFT_CONJ = {p: np.conj(sfft.fft(CA[p][_SAMPLE_CHIP])) for p in CA}


def wrap_chips(d):
    return (d + CODE_LEN / 2.0) % CODE_LEN - CODE_LEN / 2.0


def heat_rgba(z):
    z = np.clip(z, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * z - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * z - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * z - 1), 0, 1)
    return np.stack([r, g, b, np.ones_like(z)], axis=-1)


# ============================== SIGNAL MODEL ==============================
class SatSignal:
    """Truth model for one authentic satellite signal."""

    def __init__(self, prn, fd, tau0, rng):
        self.prn, self.fd, self.tau0 = prn, fd, tau0
        self.phi0 = rng.uniform(0, 2 * np.pi)
        self.bits = rng.choice([-1.0, 1.0], size=1500)   # 30 s of 50 bps nav data

    def tau(self, t):                    # code delay [chips], with code Doppler
        return self.tau0 - self.fd / CARR_PER_CHIP * t

    def code_phase(self, t):             # received code phase [chips]
        return (t * F_CHIP - self.tau(t)) % CODE_LEN


class Spoofer:
    """Carry-off spoofer: aligned start, power ramp, then a smooth accelerating code delay.

    The delay offset follows a minimum-jerk (quintic smoothstep) profile from 0 to max_off,
    whose peak acceleration equals `accel`. Starting from zero velocity and zero acceleration
    means the counterfeit peak creeps away gently first, then accelerates, then settles.
    """

    def __init__(self):
        self.active = False
        self.target = None
        self.t_launch = 0.0
        self.adv_db = 6.0     # power advantage over the true signal
        self.accel = 0.35     # peak code-delay acceleration [chips/s^2]
        self.max_off = 6.0    # final pull-off distance [chips]
        self.ramp = 1.0       # power ramp duration [s]

    @property
    def duration(self):
        """Carry-off duration [s] so that peak acceleration = accel (10*sqrt(3)/3 ≈ 5.7735)."""
        return math.sqrt(5.7735 * self.max_off / self.accel)

    @property
    def peak_rate(self):
        """Peak code-delay drift rate [chips/s]."""
        return 1.875 * self.max_off / self.duration

    def amp_factor(self, ta):
        return np.clip(ta / self.ramp, 0.0, 1.0)

    def offset(self, ta):
        td = np.clip(np.asarray(ta, dtype=float) - self.ramp, 0.0, None)
        u = np.clip(td / self.duration, 0.0, 1.0)
        return self.max_off * u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


class GnssSim:
    """Generates 1 ms blocks of complex-baseband L1 C/A signal, plus the optional spoofer."""

    def __init__(self, sat_cfg, cn0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.sats = [SatSignal(p, fd, tau, self.rng) for p, fd, tau in sat_cfg]
        self.cn0 = cn0
        self.k = 0
        self.spoof = Spoofer()
        self._n = np.arange(N_SAMP)

    @property
    def time(self):
        return self.k / FS

    def next_block(self):
        t = (self.k + self._n) / FS
        x = (self.rng.standard_normal(N_SAMP) + 1j * self.rng.standard_normal(N_SAMP)) * math.sqrt(0.5)
        A = math.sqrt(10 ** (self.cn0 / 10.0) / FS)          # noise PSD N0 = 1/FS
        sp = self.spoof
        for s in self.sats:
            code = CA[s.prn]
            tau = s.tau(t)
            d = s.bits[np.floor(t * 50).astype(np.int64) % len(s.bits)]
            ph = 2 * np.pi * s.fd * t + s.phi0
            chip = np.floor(t * F_CHIP - tau).astype(np.int64) % CODE_LEN
            x += A * d * code[chip] * np.exp(1j * ph)
            if sp.active and sp.target == s.prn:
                ta = t - sp.t_launch
                off = sp.offset(ta)
                As = A * 10 ** (sp.adv_db / 20.0) * sp.amp_factor(ta)
                chip_s = np.floor(t * F_CHIP - tau - off).astype(np.int64) % CODE_LEN
                ph_s = ph - 2 * np.pi * CARR_PER_CHIP * off   # carrier consistent with code delay
                x += As * d * code[chip_s] * np.exp(1j * ph_s)
        t0 = self.k / FS
        self.k += N_SAMP
        return x, t0


# ============================== ACQUISITION ==============================
def acquire(blocks, prn):
    """FFT parallel code-phase search; non-coherent sum over the given 1 ms blocks."""
    grid = np.zeros((len(ACQ_DOPP), N_SAMP))
    C = CODE_FFT_CONJ[prn][None, :]
    for b in blocks:
        X = sfft.fft(b[None, :] * ACQ_WIPE, axis=1)
        grid += np.abs(sfft.ifft(X * C, axis=1)) ** 2
    di, m = np.unravel_index(np.argmax(grid), grid.shape)
    peak = grid[di, m]
    row = grid[di].copy()
    row[(np.arange(-3, 4) + m) % N_SAMP] = 0.0
    ratio = peak / (row.max() + 1e-12)
    fd = ACQ_DOPP[di]
    if 0 < di < len(ACQ_DOPP) - 1:                       # parabolic Doppler refinement
        pm, pp = grid[di - 1, m], grid[di + 1, m]
        den = pm - 2 * peak + pp
        if den != 0:
            fd += 0.5 * (pm - pp) / den * (ACQ_DOPP[1] - ACQ_DOPP[0])
    a0 = math.sqrt(peak)                                 # triangle code refinement
    am, ap = math.sqrt(grid[di, (m - 1) % N_SAMP]), math.sqrt(grid[di, (m + 1) % N_SAMP])
    dm = 0.5 * (ap - am) / (a0 - min(am, ap) + 1e-12)
    loc = (-(m + dm) / 2.0) % CODE_LEN                   # code phase at block start [chips]
    return fd, loc, ratio, grid


# ============================== TRACKING ==============================
class TrackingChannel:
    def __init__(self, prn, fd, loc):
        self.prn = prn
        self.code = CA[prn]
        self.loc = loc                         # local code phase at next block start [chips]
        self.phi = 0.0
        self.vel = 2 * np.pi * fd              # PLL filter state [rad/s]
        self.acc = 0.0                         # PLL filter state [rad/s^2]
        self.w = self.vel                      # carrier NCO [rad/s]
        self.dll_acc = fd / CARR_PER_CHIP      # code Doppler [chips/s]
        self.code_rate = F_CHIP + self.dll_acc
        self.P_prev = None
        self.pli = 0.0
        self.age = 0
        self.err_t, self.err_s, self.fd_err = 0.0, None, 0.0
        self.h_t, self.h_dll, self.h_pll, self.h_err = (deque(maxlen=HIST) for _ in range(4))
        self.snaps = deque(maxlen=2)           # recent (block, loc, f) for the 3D surface
        self.last_status = ""

    def process(self, x, t0, sat, spoof):
        self.snaps.append((x, self.loc, self.w / (2 * np.pi)))
        base = self.loc + T_VEC * self.code_rate
        c = self.code
        ce = c[np.floor(base + EL_SPACING).astype(np.int64) % CODE_LEN]
        cp = c[np.floor(base).astype(np.int64) % CODE_LEN]
        cl = c[np.floor(base - EL_SPACING).astype(np.int64) % CODE_LEN]
        bb = x * np.exp(-1j * (self.phi + self.w * T_VEC))
        E, P, L = bb @ ce, bb @ cp, bb @ cl

        # --- discriminators ---
        aE, aL = abs(E), abs(L)
        dll = (1.0 - EL_SPACING) * (aE - aL) / (aE + aL + 1e-12)       # chips
        I, Q = P.real, P.imag
        pll = math.atan(Q / I) if I != 0 else math.copysign(math.pi / 2, Q)  # Costas, rad
        fe = 0.0
        if self.P_prev is not None:                                     # FLL (atan, bit-insensitive)
            z = P * np.conj(self.P_prev)
            fe = math.atan(z.imag / z.real) / T_INT if z.real != 0 else 0.0  # rad/s
        self.P_prev = P
        self.pli = 0.98 * self.pli + 0.02 * (I * I - Q * Q) / (I * I + Q * Q + 1e-12)

        # --- truth comparison (simulation only) ---
        self.err_t = wrap_chips(self.loc - sat.code_phase(t0))
        self.fd_err = self.w / (2 * np.pi) - sat.fd
        if spoof.active and spoof.target == self.prn:
            off = float(spoof.offset(t0 - spoof.t_launch))
            self.err_s = wrap_chips(self.loc - (sat.code_phase(t0) - off))
        else:
            self.err_s = None

        # --- advance NCOs ---
        self.phi = (self.phi + self.w * N_SAMP / FS) % (2 * np.pi)
        self.loc = (self.loc + self.code_rate * N_SAMP / FS) % CODE_LEN

        # --- loop filters: 3rd-order PLL + 2nd-order FLL assist, 2nd-order DLL ---
        T = T_INT
        self.acc += (W0P ** 3 * pll + W0F ** 2 * fe) * T
        self.vel += (self.acc + A3 * W0P ** 2 * pll + A2 * W0F * fe) * T
        self.w = self.vel + B3 * W0P * pll
        self.dll_acc += W0D ** 2 * dll * T
        self.code_rate = F_CHIP + self.dll_acc + A2 * W0D * dll
        self.age += 1

        self.h_t.append(t0); self.h_dll.append(dll); self.h_pll.append(pll); self.h_err.append(self.err_t)

    def status(self):
        if self.age < 300:
            s = ("PULL-IN", '#9aa0a6')
        elif self.pli < 0.4:
            s = ("LOSS OF LOCK", '#ffb020')
        elif abs(self.err_t) < 0.5 and abs(self.fd_err) < 40:
            s = ("LOCKED · TRUE", '#3ddc84')
        elif self.err_s is not None and abs(self.err_s) < 0.6:
            s = ("CAPTURED BY SPOOFER", '#ff4d4d')
        else:
            s = ("FALSE LOCK", '#ffb020')
        self.last_status = s[0]
        return s


# ============================== GUI ==============================
def dspin(lo, hi, val, step=1.0, dec=1, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val); s.setSuffix(suffix)
    return s


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNSS L1 C/A Software Receiver & Spoofing Attack Simulator")
        self.resize(1600, 980)
        self.sim, self.channels, self.curves, self.frame = None, [], [], 0
        self.surf, self.img = None, None

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        hl = QtWidgets.QHBoxLayout(central)
        hl.addWidget(self._build_controls())
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        hl.addWidget(right, 1)

        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(self._build_surface())
        top.addWidget(self._build_status())
        top.setSizes([800, 700])
        right.addWidget(top)
        right.addWidget(self._build_plots())
        right.setSizes([480, 500])

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.tick)
        self.reset()

    # ---------- construction ----------
    def _build_controls(self):
        panel = QtWidgets.QWidget(); panel.setFixedWidth(340)
        v = QtWidgets.QVBoxLayout(panel)

        g = QtWidgets.QGroupBox("True satellites (truth model)")
        grid = QtWidgets.QGridLayout(g)
        for c, h in enumerate(["PRN", "Doppler", "Code delay"]):
            grid.addWidget(QtWidgets.QLabel(f"<b>{h}</b>"), 0, c)
        self.sat_widgets = []
        for r, (prn, fd, tau) in enumerate(DEFAULT_SATS, start=1):
            sp = QtWidgets.QSpinBox(); sp.setRange(1, 32); sp.setValue(prn)
            sf = dspin(-5000, 5000, fd, 50, 0, " Hz")
            st = dspin(0, 1022.99, tau, 1, 2, " ch")
            for c, w in enumerate((sp, sf, st)):
                grid.addWidget(w, r, c)
            self.sat_widgets.append((sp, sf, st))
        v.addWidget(g)

        g2 = QtWidgets.QGroupBox("Receiver / simulation")
        f2 = QtWidgets.QFormLayout(g2)
        self.cn0 = dspin(30, 55, 47, 1, 1, " dB-Hz"); f2.addRow("C/N0", self.cn0)
        self.speed = QtWidgets.QSpinBox(); self.speed.setRange(1, 40); self.speed.setValue(10)
        self.speed.setSuffix(" ms / frame"); f2.addRow("Sim speed", self.speed)
        self.btn_reset = QtWidgets.QPushButton("⟲  Reset && Re-acquire"); self.btn_reset.clicked.connect(self.reset)
        self.btn_pause = QtWidgets.QPushButton("⏸  Pause"); self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self.toggle_pause)
        f2.addRow(self.btn_reset); f2.addRow(self.btn_pause)
        v.addWidget(g2)

        g3 = QtWidgets.QGroupBox("Spoofer (carry-off attack)")
        f3 = QtWidgets.QFormLayout(g3)
        self.target = QtWidgets.QComboBox(); f3.addRow("Target / 3D view", self.target)
        self.adv = dspin(0, 12, 6, 0.5, 1, " dB"); f3.addRow("Power advantage", self.adv)
        self.accel = dspin(0.05, 2.0, 0.35, 0.05, 2, " chip/s²"); f3.addRow("Peak delay accel.", self.accel)
        self.maxoff = dspin(0.5, 14, 6, 0.5, 1, " chips"); f3.addRow("Pull-off distance", self.maxoff)
        self.ramp = dspin(0.2, 5, 1.0, 0.1, 1, " s"); f3.addRow("Power ramp", self.ramp)
        self.btn_attack = QtWidgets.QPushButton("🚀  Launch Spoofing Attack")
        self.btn_attack.setMinimumHeight(46)
        self._style_attack(False)
        self.btn_attack.clicked.connect(self.toggle_attack)
        f3.addRow(self.btn_attack)
        v.addWidget(g3)

        info = QtWidgets.QLabel(
            "<small><b>How it works:</b> The spoofer starts code- and phase-aligned with the target PRN, "
            "so the receiver can't tell the two apart. It ramps its power above the true signal, then "
            "drags its code delay with a smooth acceleration while keeping its carrier consistent. The "
            "DLL/PLL follow the stronger, counterfeit peak, and the true peak slides away in the 3D "
            "view.<br><br>Stopping the attack mid-way removes the counterfeit peak and the loops drop "
            "into a false lock. Use <i>Reset &amp; Re-acquire</i> to start again.<br><br>"
            "Lower the power advantage (e.g. 3 dB) to see a weaker spoofer struggle and break lock."
            "<br><br>1 chip ≈ 293 m of pseudorange error.</small>")
        info.setWordWrap(True)
        v.addWidget(info)
        v.addStretch(1)
        return panel

    def _build_surface(self):
        box = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(box); vb.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("<b>Real-time 3D correlation surface</b> — X: code delay offset [chips, ±15] · "
                                 "Y: Doppler offset [±2.5 kHz] · Z: normalized |R|²  (centre = receiver's tracking point)")
        title.setWordWrap(True)
        vb.addWidget(title)
        nx, ny = len(VIEW_CODE), len(VIEW_DOPP)
        z0 = np.zeros((nx, ny))
        if HAVE_GL:
            self.glw = gl.GLViewWidget()
            try:
                self.glw.setBackgroundColor('k')
            except Exception:
                pass
            self.glw.setCameraPosition(distance=55, elevation=28, azimuth=-55)
            grid = gl.GLGridItem(); grid.setSize(30, 30); grid.setSpacing(5, 5)
            self.glw.addItem(grid)
            ax = gl.GLAxisItem(); ax.setSize(16, 16, 13); self.glw.addItem(ax)
            # NOTE: no `colors=` here -- colours are set as a flat array in _set_surface()
            self.surf = gl.GLSurfacePlotItem(x=VIEW_CODE / 2.0, y=VIEW_DOPP / 1000.0 * 6.0, z=z0,
                                             smooth=False)
            self.glw.addItem(self.surf)
            self._set_surface(z0)
            vb.addWidget(self.glw, 1)
        else:
            pw = pg.PlotWidget(title="(PyOpenGL not found — 2D heat-map fallback)")
            pw.setLabel('bottom', 'code delay offset [chips]'); pw.setLabel('left', 'Doppler offset [kHz]')
            self.img = pg.ImageItem()
            lut = (heat_rgba(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            self.img.setLookupTable(lut)
            self.img.setRect(QtCore.QRectF(-15, -2.5, 30, 5))
            pw.addItem(self.img)
            vb.addWidget(pw, 1)
        self.surf_caption = QtWidgets.QLabel("")
        self.surf_caption.setWordWrap(True)
        vb.addWidget(self.surf_caption)
        return box

    def _build_status(self):
        box = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(box); vb.setContentsMargins(0, 0, 0, 0)
        self.banner = QtWidgets.QLabel("")
        self.banner.setAlignment(QtCore.Qt.AlignCenter); self.banner.setWordWrap(True)
        self.banner.setMinimumHeight(48)
        vb.addWidget(self.banner)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["PRN", "Est. Doppler [Hz]", "True Doppler [Hz]",
                                              "Code phase [chip]", "Δτ vs TRUE [chip]", "PLI", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        vb.addWidget(self.table, 1)
        self.logbox = QtWidgets.QPlainTextEdit(); self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(300)
        f = QtGui.QFont("Monospace"); f.setStyleHint(QtGui.QFont.TypeWriter); self.logbox.setFont(f)
        vb.addWidget(self.logbox, 1)
        return box

    def _build_plots(self):
        w = pg.GraphicsLayoutWidget()
        self.p_dll = w.addPlot(row=0, col=0, title="DLL discriminator — normalized E−L envelope [chips]")
        self.p_pll = w.addPlot(row=1, col=0, title="PLL discriminator — Costas atan(Q/I) [rad]")
        self.p_err = w.addPlot(row=2, col=0, title="Code tracking error w.r.t. TRUE signal [chips]  (pull-off distance)")
        for p in (self.p_dll, self.p_pll, self.p_err):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.addLegend(offset=(10, 5))
            p.setDownsampling(mode='peak'); p.setClipToView(True)
        self.p_pll.setXLink(self.p_dll); self.p_err.setXLink(self.p_dll)
        self.p_dll.setYRange(-0.6, 0.6); self.p_pll.setYRange(-1.7, 1.7)
        self.p_err.setLabel('bottom', 'receiver time [s]')
        return w

    # ---------- helpers ----------
    def _set_surface(self, Z):
        """Update surface height + per-vertex colours.

        Colours go to MeshData as a flat (nx*ny, 4) array matching the row-major vertex order.
        This avoids 'IndexError: index 61 is out of bounds for axis 0 with size 61' on pyqtgraph
        versions that do not reshape a (nx, ny, 4) colour array.
        """
        if self.surf is None:
            return
        self.surf.setData(z=Z * 12.0)
        cols = heat_rgba(Z).reshape(-1, 4).astype(np.float32)
        md = getattr(self.surf, '_meshdata', None)
        if md is None:
            md = getattr(self.surf, 'opts', {}).get('meshdata')
        if md is not None:
            md.setVertexColors(cols)
            self.surf.meshDataChanged()

    def log(self, msg):
        t = self.sim.time if self.sim else 0.0
        self.logbox.appendPlainText(f"[{t:7.3f}s] {msg}")

    def _style_attack(self, active):
        if active:
            self.btn_attack.setText("■  Stop Spoofing Attack")
            self.btn_attack.setStyleSheet("background:#444;color:#ffb3b3;font-weight:bold;font-size:14px;")
        else:
            self.btn_attack.setText("🚀  Launch Spoofing Attack")
            self.btn_attack.setStyleSheet("background:#b00020;color:white;font-weight:bold;font-size:14px;")

    def _mark(self, t, color, label=None):
        for p in (self.p_dll, self.p_pll, self.p_err):
            p.addItem(pg.InfiniteLine(pos=t, angle=90, label=label,
                                      pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.DashLine),
                                      labelOpts={'color': color, 'position': 0.9}))

    def _target_channel(self):
        prn = self.target.currentData()
        for ch in self.channels:
            if ch.prn == prn:
                return ch
        return None

    # ---------- actions ----------
    def reset(self):
        self.timer.stop()
        cfg = [(sp.value(), sf.value(), st.value()) for sp, sf, st in self.sat_widgets]
        if len({c[0] for c in cfg}) != len(cfg):
            QtWidgets.QMessageBox.warning(self, "Config", "PRNs must be unique.")
            return
        self.sim = GnssSim(cfg, self.cn0.value())
        self.logbox.clear()
        self.log(f"Acquisition: FFT parallel code search, {len(ACQ_DOPP)} Doppler bins "
                 f"({ACQ_DOPP[1] - ACQ_DOPP[0]:.0f} Hz), 2 ms non-coherent")
        blocks = [self.sim.next_block()[0] for _ in range(2)]
        self.channels = []
        for s in self.sim.sats:
            fd, loc0, ratio, _ = acquire(blocks, s.prn)
            self.log(f"PRN {s.prn:2d}: fd={fd:+8.1f} Hz (true {s.fd:+.0f}) | code={loc0:7.2f} ch "
                     f"(true {s.code_phase(0.0):7.2f}) | peak ratio {ratio:5.1f}")
            loc = (loc0 + 2 * N_SAMP * (F_CHIP + fd / CARR_PER_CHIP) / FS) % CODE_LEN
            self.channels.append(TrackingChannel(s.prn, fd, loc))
        self.log("Tracking: 2nd-order DLL (Bn=2 Hz) + 3rd-order Costas PLL (Bn=18 Hz) with FLL assist (Bn=8 Hz)")

        self.target.clear()
        for ch in self.channels:
            self.target.addItem(f"PRN {ch.prn}", ch.prn)

        self.table.setRowCount(len(self.channels))
        for r in range(len(self.channels)):
            for c in range(7):
                it = QtWidgets.QTableWidgetItem("")
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, c, it)

        self.curves = []
        for p in (self.p_dll, self.p_pll, self.p_err):
            p.clear()
        for ch, col in zip(self.channels, CH_COLORS):
            pen = pg.mkPen(col, width=1)
            self.curves.append(tuple(p.plot([], [], pen=pen, name=f"PRN {ch.prn}")
                                     for p in (self.p_dll, self.p_pll, self.p_err)))
        self._style_attack(False)
        self.frame = 0
        if not self.btn_pause.isChecked():
            self.timer.start()

    def toggle_pause(self, paused):
        self.btn_pause.setText("▶  Run" if paused else "⏸  Pause")
        if paused:
            self.timer.stop()
        elif self.sim is not None:
            self.timer.start()

    def toggle_attack(self):
        if self.sim is None:
            return
        sp = self.sim.spoof
        t = self.sim.time
        if not sp.active:
            sp.target = self.target.currentData()
            sp.adv_db, sp.accel = self.adv.value(), self.accel.value()
            sp.max_off, sp.ramp = self.maxoff.value(), self.ramp.value()
            sp.t_launch, sp.active = t, True
            self._style_attack(True)
            self._mark(t, '#ff4d4d', "ATTACK")
            self.log(f"SPOOFER ON  -> PRN {sp.target}: +{sp.adv_db:.1f} dB, ramp {sp.ramp:.1f}s, "
                     f"peak accel {sp.accel:.2f} ch/s², pull-off {sp.max_off:.1f} ch in {sp.duration:.1f}s "
                     f"(peak drift {sp.peak_rate:.2f} ch/s ≈ {sp.peak_rate * CARR_PER_CHIP:.0f} Hz)")
        else:
            sp.active = False
            self._style_attack(False)
            self._mark(t, '#bbbbbb', "stop")
            self.log("SPOOFER OFF")

    # ---------- main loop ----------
    def tick(self):
        if self.sim is None:
            return
        self.sim.cn0 = self.cn0.value()
        for _ in range(self.speed.value()):
            x, t0 = self.sim.next_block()
            for ch, sat in zip(self.channels, self.sim.sats):
                ch.process(x, t0, sat, self.sim.spoof)
        self.frame += 1
        self.update_plots()
        self.update_table()
        if self.frame % 3 == 0:
            self.update_surface()

    def update_plots(self):
        for ch, (cd, cp, ce) in zip(self.channels, self.curves):
            t = np.array(ch.h_t)
            cd.setData(t, np.array(ch.h_dll))
            cp.setData(t, np.array(ch.h_pll))
            ce.setData(t, np.array(ch.h_err))

    def update_table(self):
        prev = {ch.prn: ch.last_status for ch in self.channels}
        for r, (ch, sat) in enumerate(zip(self.channels, self.sim.sats)):
            st, col = ch.status()
            vals = [f"{ch.prn}", f"{ch.w / (2 * np.pi):+.1f}", f"{sat.fd:+.1f}", f"{ch.loc:.2f}",
                    f"{ch.err_t:+.3f}", f"{ch.pli:+.2f}", st]
            for c, v in enumerate(vals):
                self.table.item(r, c).setText(v)
            self.table.item(r, 6).setForeground(QtGui.QBrush(QtGui.QColor(col)))
            if prev[ch.prn] and prev[ch.prn] != st:
                self.log(f"PRN {ch.prn:2d}: {prev[ch.prn]} -> {st}")
        sp = self.sim.spoof
        cap = [ch for ch in self.channels if ch.last_status == "CAPTURED BY SPOOFER"]
        if cap:
            ch = cap[0]
            self.banner.setText(f"⚠  PRN {ch.prn}: DLL/PLL CAPTURED — pulled {abs(ch.err_t):.2f} chips "
                                f"(≈ {abs(ch.err_t) * CHIP_M:,.0f} m pseudorange error) off the TRUE signal")
            self.banner.setStyleSheet("background:#4a0000;color:#ff8a8a;font-size:15px;font-weight:bold;")
        elif sp.active:
            self.banner.setText(f"Spoofer transmitting on PRN {sp.target} — power ramp / carry-off in progress…")
            self.banner.setStyleSheet("background:#4a3500;color:#ffd27a;font-size:15px;font-weight:bold;")
        elif any(ch.last_status in ("LOSS OF LOCK", "FALSE LOCK") for ch in self.channels):
            self.banner.setText("Loss of lock — reset & re-acquire to recover")
            self.banner.setStyleSheet("background:#4a3500;color:#ffd27a;font-size:15px;font-weight:bold;")
        else:
            self.banner.setText("✓  All channels tracking authentic signals")
            self.banner.setStyleSheet("background:#003d1f;color:#7dffb5;font-size:15px;font-weight:bold;")

    def update_surface(self):
        ch = self._target_channel()
        if ch is None or not ch.snaps:
            return
        Z = np.zeros((len(VIEW_CODE), len(VIEW_DOPP)))
        C = CODE_FFT_CONJ[ch.prn][None, :]
        for blk, loc, f in list(ch.snaps):
            wipe = VIEW_WIPE * np.exp(-2j * np.pi * f * T_VEC)[None, :]
            r = np.abs(sfft.ifft(sfft.fft(blk[None, :] * wipe, axis=1) * C, axis=1)) ** 2
            mc = int(round(-2.0 * loc)) % N_SAMP
            Z += r[:, (mc + VIEW_CODE) % N_SAMP].T
        Z /= Z.max() + 1e-12
        if HAVE_GL:
            self._set_surface(Z)
        else:
            self.img.setImage(Z, levels=(0, 1), autoLevels=False)
        txt = (f"PRN {ch.prn}: the centre peak is what the receiver is tracking. The TRUE signal is at "
               f"Δτ = {ch.err_t:+.2f} chips, Δf = {-ch.fd_err:+.0f} Hz relative to it.")
        if ch.last_status == "CAPTURED BY SPOOFER":
            txt = "<span style='color:#ff6b6b'><b>Tracking the COUNTERFEIT peak.</b></span> " + txt
        self.surf_caption.setText(txt)


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, background='k', foreground='w')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
