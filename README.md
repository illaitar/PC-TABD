# PC-TABD / PC-TAB
**Physically Consistent Trajectory-Aware Blur**: a physically grounded motion-blur synthesis pipeline and a trajectory-aware dataset.

## Timeline


---

## Overview
This repository contains code and materials for **PC-TAB**, a physically grounded motion blur synthesis framework that:
- approximates the exposure integral **in linear light**,
- uses **three consecutive sharp frames** as anchors (prev / reference / next),
- decomposes motion into **camera + object residual**, accounts for **visibility/occlusions**, **rolling-shutter effects**, and a lightweight **ISP simulation** (e.g., noise/clamp).

Built on top of this pipeline, we provide **PC-TABD** — a synthetic blur–sharp dataset with **explicit per-pixel trajectories**, along with auxiliary signals such as depth/flow and generation metadata.

---

## Why
The goal is to reduce the synthetic–real gap when training or fine-tuning deblurring models. Instead of simple frame averaging, we aim to reproduce key factors of real-world blur formation (exposure, trajectory shapes, rolling shutter, occlusions, ISP), while keeping the process controllable and interpretable.

---

## Dataset link
**PC-TABD dataset (Yandex Disk):**  
**https://disk.yandex.ru/d/ZWUnAKowUmQSzg**

> This link points to the dataset download (archives / folder structure with blur–sharp pairs, trajectories, and metadata).

---

## Repository structure (high-level)
> Folder names may slightly differ; below is the intended “mental map”.

- `pc_tab/` — core synthesis library (trajectories → visibility → integration → ISP)
- `configs/` — YAML/JSON configs for generation/experiments
- `scripts/`
  - `generate_dataset.py` — generate PC-TABD samples
  - `augment_train.py` — on-the-fly augmentation during training
  - `evaluate.py` — evaluation / metrics runner
- `third_party/` — external dependencies / wrappers (e.g., flow/depth tools)
- `docs/` — notes, figures, additional materials
- `assets/` — README images
- `data/` — (optional) place for dataset symlinks/caches (do not commit large files)

---

## Quickstart

### 1) Installation
```bash
git clone https://github.com/illaitar/PC-TABD
cd PC-TABD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download the dataset
Download from the Yandex Disk link and unpack, e.g. into `./data/pc_tabd/`:
```bash
mkdir -p data/pc_tabd
# unzip / tar / etc.
```

### 3) Generate blur samples (example)
```bash
python scripts/generate_dataset.py \
  --config configs/pc_tabd.yaml \
  --out data/pc_tabd_generated
```

---

## Citation
TODO

---

## License
This project is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.
