# MMA3001 — Automated Pork Rasher Packaging Inspection

A numerical methods and machine learning project for detecting packaging defects in pork rasher products using YOLOv8n and spatial finite differences.

**Unit:** MMA3001 — Numerical Methods and Machine Learning  
**University:** Monash University  
**Author:** Quoc Hung Vo  
**Language:** Python 3.12

## 1. Engineering Problem

Packaging defects in food manufacturing can compromise product quality and packaging integrity. Manual visual inspection is time-consuming and can be inconsistent.

This project investigates an automated, image-based inspection approach that identifies and localises visible defects in pork rasher packaging.

The system detects five classes:

- Loose meat
- Packaging error
- Twisted meat
- Unsealed packaging
- Wrinkle

The intended application is to support preliminary quality inspection rather than replace validated industrial inspection systems.

## 2. Computational Approach

Two computational approaches are investigated.

**Machine Learning — YOLOv8n**

A pretrained YOLOv8n object detection model is fine-tuned to identify and localise packaging defects using bounding boxes.

**Numerical Methods — Finite Differences**

Spatial finite differences are implemented to calculate image intensity gradients. Analytically known derivatives are used to verify numerical accuracy and convergence.

Gaussian blur experiments investigate changes in image edge strength and the corresponding detection performance of YOLOv8n.

## 3. Dataset

**Source:** [Pork Rasher Error-Packaging — Roboflow Universe, Version 4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4)

The supplied dataset contains 3,549 images in YOLO detection format.

| Split | Original | Cleaned |
|---|---:|---:|
| Training | 3,349 | 3,313 |
| Validation | 120 | 119 |
| Test | 80 | 80 |
| Total | 3,549 | 3,512 |

Dataset auditing detected 13 source IDs shared across splits. A deterministic cleaning procedure applied the priority `test > validation > train`, removing 37 potentially overlapping images while preserving all test images.

The cleaning procedure retains valid empty annotation files.

The dataset is not included in this Git repository. Users must obtain the source dataset separately and follow its applicable licensing and attribution requirements.

## 4. Installation

Tested environment:

- Windows 11
- Python 3.12
- NVIDIA RTX 4050 Laptop GPU
- PyTorch 2.11.0+cu128
- Ultralytics 8.4.190

Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The requirements are configured for CUDA 12.8. A compatible NVIDIA GPU and driver are required by the GPU-only training and evaluation scripts.

## 5. Reproducing the Workflow

Run all commands from the repository root.

**Step 1 — Dataset Audit**

Place the downloaded Version 4 ZIP inside `data/raw/` using the filename expected by the scripts.

```powershell
python src/data_audit.py
```

Audit results are saved to `results/data_audit.json`.

**Step 2 — Dataset Cleaning**

```powershell
python -m src.data_cleaning --output data/cleaned/v1
```

The cleaned dataset and its `data.yaml` are generated under `data/cleaned/v1/`.

**Step 3 — Automated Tests**

```powershell
python -m pytest tests/test_phase1.py -q
```

The verified development run passed 40 tests.

**Step 4 — YOLO Training**

Optional one-epoch smoke test:

```powershell
python -m src.train_yolo --mode smoke
```

Main training:

```powershell
python -m src.train_yolo --mode main
```

Main training uses 25 epochs, early stopping patience 7, 640-pixel input size and batch size 4.

Training outputs are stored under `runs/detect/`.

**Step 5 — Finite Difference Verification**

```powershell
python -m src.finite_difference --mode verify
```

Analyse validation images:

```powershell
python -m src.finite_difference --mode analyze
```

**Step 6 — Blur Robustness**

```powershell
python -m src.blur_robustness
```

**Step 7 — Inference Resolution Comparison**

```powershell
python -m src.optimise_inference
```

**Step 8 — Final Test Evaluation**

```powershell
python -m src.final_evaluation
```

The last three evaluation scripts expect the trained checkpoint at `runs/detect/main-1eqhuzvr/weights/best.pt`. If training produces a differently named experiment directory, update the checkpoint paths before running them.

The trained checkpoint is currently excluded from Git tracking. It can be reproduced through the training workflow.

## 6. Main Experimental Results

### Finite Differences

Numerical verification passed all 10 checks.

- Interior centred differences demonstrated approximately second-order convergence.
- Forward and backward boundary differences demonstrated approximately first-order convergence.

Mean image gradient magnitude decreased by 24.319% at Gaussian blur σ = 1 and 49.944% at σ = 2.

### YOLO Blur Robustness

| Blur σ | mAP@50 | mAP@50–95 |
|---|---:|---:|
| 0 | 0.5551 | 0.3310 |
| 1 | 0.4874 | 0.2784 |
| 2 | 0.4830 | 0.2830 |

These results show that detection performance changes under image blur, although the relationship is not strictly monotonic for every metric.

### Inference Resolution Optimisation

| Resolution | mAP@50 | mAP@50–95 | Inference (ms/image) |
|---|---:|---:|---:|
| 480 | 0.5296 | 0.2804 | 2.0788 |
| 640 | 0.5551 | 0.3310 | 2.9033 |
| 800 | 0.3988 | 0.2289 | 3.9283 |

The 640-pixel configuration was selected before final test evaluation because it provided the highest validation accuracy among the three tested resolutions.

Reported inference timing excludes preprocessing and postprocessing and is subject to GPU measurement variability.

### Final Withheld Test Evaluation

| Metric | Result |
|---|---:|
| Test images | 80 |
| Ground-truth boxes | 79 |
| Precision | 0.3364 |
| Recall | 0.4766 |
| F1 | 0.3944 |
| mAP@50 | 0.4609 |
| mAP@50–95 | 0.2365 |

The final test configuration was fixed before examining test results.

## 7. Limitations

- The dataset has a substantial class imbalance.
- Some classes have very few validation and test annotations.
- Wrinkle has no ground-truth annotations in either validation or test, preventing a meaningful evaluation of that class.
- Filename-based source separation cannot guarantee the absence of all visually or temporally related images.
- Numerical edge strength is not a direct measure of object detection accuracy.
- GPU inference timings are not equivalent to industrial end-to-end inspection latency.
- The model has not been validated for real production deployment.

## 8. Project Structure

- `src/` — Python implementations
- `tests/` — Automated verification tests
- `data/` — Local dataset storage
- `results/` — Numerical and experimental outputs
- `runs/` — Local training results and checkpoints
- `docs/` — Source code documentation
- `reports/` — Project documentation

## 9. AI Assistance

ChatGPT and OpenAI Codex were used to assist with project planning, code generation, debugging, documentation and result interpretation.

Generated code was manually transferred into VS Code and verified through automated tests, numerical verification, GPU training and experimental evaluation.

A fuller discussion of AI contributions, errors, verification and student decision-making is included in the project report.

## 10. Licensing and References

See the repository `LICENSE` file for the licence applicable to student-authored source code. Dataset, pretrained model and third-party software are governed by their respective licences.

Key resources:

- MMA3001 teaching materials, Monash University, 2026.
- Roboflow Universe, Pork Rasher Error-Packaging Dataset, Version 4.
- Ultralytics, YOLOv8 documentation and pretrained model.
- PyTorch documentation.
