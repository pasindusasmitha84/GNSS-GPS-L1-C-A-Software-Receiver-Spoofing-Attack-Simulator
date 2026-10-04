# GNSS / GPS L1 C/A Software Receiver & Spoofing Attack Simulator

A comprehensive single-file PyQt5, PyQtGraph, and SciPy desktop application that simulates a live GPS L1 C/A software receiver and models sophisticated spoofing carry-off attacks in real time.

---

## 🛰️ Key Features

* **Synthetic Baseband Signal Generation:** Simulates complex-baseband GPS L1 C/A signals across 3 configurable PRNs with customizable Doppler shifts, code delays, 50 bps navigation data, and additive white Gaussian noise (AWGN) at adjustable C/N0 levels ($f_s = 2.046 \text{ MHz}$).
* **Parallel Code-Phase Acquisition:** Implements FFT-based parallel code-phase searching across acquisition Doppler bins, complete with parabolic Doppler and code refinements.
* **Advanced Tracking Loops:** Features a 2nd-order Delay Lock Loop (DLL) using a normalized early-minus-late envelope and a 3rd-order Costas Phase Lock Loop (PLL) backed by continuous 2nd-order Frequency Lock Loop (FLL) assistance to maintain stable tracking[cite: 8].
* **Code- & Carrier-Consistent Spoofer:** Models a sophisticated spoofer that initiates phase alignment with a target PRN, ramps power above the authentic signal, and executes a smooth minimum-jerk (quintic smoothstep) accelerating code-delay offset to drag the receiver's tracking loops off-target[cite: 8].
* **Real-Time 3D Correlation Surface & Diagnostics:** Features an interactive 3D correlation surface (code delay vs. Doppler) rendered via PyOpenGL (with a 2D heat-map fallback), live DLL/PLL discriminator plots, tracking error metrics versus truth, and an updating lock-status table[cite: 8].

---

## 🛠️ Tech Stack

* **Core & Math:** Python[cite: 8], NumPy[cite: 8], SciPy (FFT module)[cite: 8]
* **GUI & Plotting:** PyQt5[cite: 8], PyQtGraph[cite: 8]
* **3D Visualization:** PyOpenGL[cite: 8]

---

## 📦 Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
