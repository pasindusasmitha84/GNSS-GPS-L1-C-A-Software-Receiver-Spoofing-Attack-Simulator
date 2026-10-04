# 5G NR Massive MIMO — RL Beamforming & Anti-Jamming Simulator

An interactive, real-time desktop simulation modeling a 5G gNB equipped with a 64-element Uniform Linear Array (ULA) operating in the n78 band (3.5 GHz) to maintain robust uplink communications under dynamic rogue RF jamming.

---

## 🚀 Key Features

* **Dynamic RF Environment:** Models 2 mobile legitimate UEs and 1 rogue RF jammer featuring random-walk mobility and an intelligent pursuit behavior mode targeting active UEs.
* **Model-Based Deep RL Agent:** Utilizes a PyTorch-backed actor network that ingests Direction of Arrival (DOA) estimates and received power metrics to output complex antenna weights parameterised in the signal subspace.
* **Policy-Guided L-BFGS Planner:** Refines actor policy outputs frame-by-frame via an L-BFGS optimizer on a differentiable sum-rate reward model ($\sum_k \log_2(1 + \text{SINR}_k)$), enabling deep null-steering against agile jammers.
* **Robust Benchmarking:** Directly compares real-time performance against classical Minimum Variance Distortionless Response (MVDR / Capon) and conventional matched-filter beamforming using identical noisy channel estimates[cite: 6].
* **Interactive PyQt5 Dashboard:** Features a live 2D geographic deployment map, polar antenna radiation pattern visualizers, and rolling post-beamforming SINR history plots[cite: 6].

---

## 🛠️ Tech Stack

* **Core & Math:** Python[cite: 6], NumPy[cite: 6]
* **Machine Learning & Optimization:** PyTorch (with CUDA acceleration support)[cite: 6], L-BFGS optimization[cite: 6]
* **GUI & Visualization:** PyQt5[cite: 6], PyQtGraph[cite: 6]

---

## 📦 Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
