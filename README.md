# PlasticityX

**A Multi-Dimensional Computational Framework for Resolving Tumor-Intrinsic Plasticity and Simulating 3D Quasi-Potential Landscapes**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)

---

## Overview

PlasticityX is an open-source computational biology and software engineering platform designed to resolve tumor-intrinsic cellular plasticity (TIP) from high-dimensional bulk transcriptomics. It resolves the fundamental mathematical limitations of 1D scalar EMT metrics by:
1. **Bi-Axial State Mapping:** Quantifying orthogonal directional EMT ($d_{\text{EMT}}$) and Cancer Stem Cell ($s_{\text{CSC}}$) coordinates.
2. **Entropy-Guided Decontamination:** Implementing an adaptive Shannon entropy attenuation factor ($\alpha$) that filters non-malignant fibroblast/stromal contamination.
3. **Continuous 3D Quasi-Potential Landscapes:** Simulating cellular state stability across an analytical double-well potential surface $U(X,Y)$ rendered in real time using Plotly WebGL.
4. **Closed-Loop Oncology Analytics:** Providing GDSC2-calibrated pharmacogenomic sensitivity predictions ($\ln(\text{IC}_{50})$) and discrete step Kaplan-Meier survival curves with log-rank testing.

---

## Authors

* **Bavin** (Lead Author, Architecture & Methodology)
* **Sermaraja** (Validation, Testing & Data Curation)

---

## Quickstart & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/gbavin2005-glitch/PlasticityX.git](https://github.com/gbavin2005-glitch/PlasticityX.git)
cd PlasticityX
