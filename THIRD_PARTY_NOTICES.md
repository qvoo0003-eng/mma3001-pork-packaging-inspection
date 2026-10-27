# Third-party software, model and dataset notices

This repository is an academic MMA3001 project and uses external software, a pretrained object detector and a third-party image dataset.

## Ultralytics YOLOv8

- Software and pretrained model: **Ultralytics YOLOv8n**.
- Source and documentation: https://github.com/ultralytics/ultralytics and https://docs.ultralytics.com/models/yolov8/
- Licensing: Ultralytics provides its YOLO code and trained models under **AGPL-3.0** by default, with separate Enterprise licensing available. See https://www.ultralytics.com/license.
- `runs/detect/main-1eqhuzvr/weights/best.pt` is a fine-tuned YOLOv8n checkpoint derived from Ultralytics pretrained weights. It is not a separately developed model architecture.

The repository's `LICENSE` contains the GNU Affero General Public License version 3. The repository license does **not** relicense independently copyrighted training data or unrelated third-party packages.

## Pork Rasher Error-Packaging dataset

- Source: Roboflow Universe, **Pork Rasher Error-Packaging**, Version 4.
- Dataset page: https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4
- The dataset's bundled `data.yaml` identifies the dataset license as **CC BY 4.0**. See https://creativecommons.org/licenses/by/4.0/.
- The full dataset is **not redistributed** in this Git repository. The README gives the source and steps for local preparation. Dataset ownership and attribution remain with the dataset's rights holders.

## Other dependencies

Python, PyTorch, Torchvision, OpenCV, NumPy, Matplotlib, PyYAML, pytest, pdoc and their dependencies have their own respective third-party licenses. See each upstream project for licensing terms.

## AI use

ChatGPT and Codex assisted with planning, code generation, documentation and troubleshooting. The student manually ran and checked the project, numerical verification, training, experiments and final evaluation. See the written report for the detailed AI-use reflection.
