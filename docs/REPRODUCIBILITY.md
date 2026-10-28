# Reproducibility and dataset access

This note complements the [project README](../README.md). All commands below are run from the repository root in PowerShell unless stated otherwise. The dataset itself is not redistributed in this repository.

## Known-good environment

- Windows 11, Python 3.12.10, NVIDIA RTX 4050 Laptop GPU (approximately 6 GB VRAM)
- PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; Ultralytics 8.4.190
- Installation: `python -m venv .venv`, activate with `.\.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`
- The pinned environment was verified with `pip install --dry-run` on the original development machine; a clean-machine installation was **not** independently tested. Training/evaluation scripts require a CUDA-capable GPU and do not automatically fall back to CPU.

## Getting the input dataset

1. Download **Pork Rasher Error-Packaging — Version 4 (YOLOv8 format)** from [Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4). Follow Roboflow's current access/download instructions and the dataset's applicable licence.
2. Create the folder if it is missing: `New-Item -ItemType Directory -Force data/raw | Out-Null`.
3. Place the **unextracted** archive at this exact path (the scripts' default):

   `data/raw/Pork Rasher Error-Packaging-.v4i.yolov8.zip`

4. To check that the archive matches the one used for this project, run:

   `Get-FileHash 'data/raw/Pork Rasher Error-Packaging-.v4i.yolov8.zip' -Algorithm SHA256`

   Verified original SHA-256:
   `88CC5B2527DB50DC6E39D3D8C8379D1B23A6D8C7DD105303EF6551347DDBC66D`

   If Roboflow supplies a newer/repacked archive or changes the filename, hashes may differ; do not claim exact replication without verifying its content.

## Run the workflow

```powershell
python src/data_audit.py
python -m src.data_cleaning --output data/cleaned/v1
python -m pytest tests/test_phase1.py -q
python -m src.finite_difference --mode verify
python -m src.finite_difference --mode analyze
```

The cleaning command deliberately refuses an output directory that already exists: on a second run, choose a new output name or review the existing output manually. Do not delete or overwrite the original archive.

Reference counts from the verified version-4 archive:

| Split | Original images | Cleaned images |
|---|---:|---:|
| Train | 3349 | 3313 |
| Validation | 120 | 119 |
| Test | 80 | 80 |

There were 13 source IDs shared across splits before cleaning. Cleaning used `test > valid > train` priority and removed 37 images while keeping the test set unchanged. These checks address *recognised filename-level source IDs*; they cannot guarantee independence from visual/temporal near-duplicates.

## Pretrained model and experiment scripts

The exact fine-tuned checkpoint used in the reported experiments is **included** in this Git repository at:

`runs/detect/main-1eqhuzvr/weights/best.pt`

SHA-256:
`DA78E09E684D1AB2AF149D7D99E0B08477028632D9D3A57ACE97FB3B4C3901EE`

This checkpoint was verified byte-for-byte against a fresh Git clone, and the clone passed 40/40 Phase-1 automated tests using the existing local Python environment. Test results do not amount to independent verification of a fresh environment installation.

To run the trained-model experiments after preparing the cleaned dataset:

```powershell
python -m src.blur_robustness
python -m src.optimise_inference
python -m src.final_evaluation
```

These scripts expect the checkpoint at the fixed path above. The same pretrained YOLOv8n architecture can also be fine-tuned using `python -m src.train_yolo --mode main`, but a new training run is not guaranteed to reproduce bit-identical weights, timings or metrics. If rerunning the cleaning or training workflow, avoid reselecting a configuration based on the withheld test results.

## Verification records and limitations

- Automated test report: [`reports/test_report.txt`](../reports/test_report.txt)
- Finite-difference verification output: [`reports/evidence/finite_difference_verification.txt`](../reports/evidence/finite_difference_verification.txt)
- CSV evidence and selected plots: [`reports/evidence/`](../reports/evidence/)
- Generated API documentation: [`docs/api/index.html`](api/index.html)
- Official code and model attribution: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)

`reports/evidence` stores selected report-ready summaries, not every local Ultralytics output. The complete original ZIP, cleaned images and intermediate training/validation caches are excluded due to size and third-party data licensing considerations. The final report PDF is added separately once completed.
