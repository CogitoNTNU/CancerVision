<!-- TODO: CHANGE ALL INSTANCES OF "PROJECT-TEMPLATE" IN ENTIRE PROJECT TO YOUR PROJECT TITLE-->

# CancerVision

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/PROJECT-TEMPLATE/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/PROJECT-TEMPLATE)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/PROJECT-TEMPLATE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/project-logo.webp" width="50%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b> Table of contents </b></summary>

- [PROJECT-TEMPLATE](#PROJECT-TEMPLATE)
  - [Description](#description)
  - [Prerequisites](#%EF%B8%8F-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->
CancerVision is an AI project that uses computer vision and deep learning to segment brain tumors from MRI scans.
It identifies tumor regions such as the core, edema, and enhancing areas to support medical analysis and treatment planning.
The aim is to build accurate models that reduce manual work and improve consistency in diagnosis.



## Prerequisites

<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.  

-->

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)  
-->

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/CancerVision
   cd CancerVision
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

1. **Resources**:
   - https://www.slicer.org/
   - https://zenodo.org/records/11199559
   - https://github.com/SverreNystad/computer-vision/blob/main/docs/TDT4265_project_2025_v1.0.pdf


<!--
1. **Configure environment variables**:
    This project uses environment variables for configuration. Copy the example environment file to create your own:
    ```sh
    cp .env.example .env
    ```
    Then edit the `.env` file to include your specific configuration settings.
-->

1. **Set up pre commit** (only for development):
   ```sh
   uv run pre-commit install
   ```

## Usage


You can run the project using either Python directly or with uv. Below are examples for both methods:

**Using Python:**
```bash
python main.py datasets
python main.py datasets --search-dir /path/to/data --max-depth 5
python main.py models
python main.py web --host 127.0.0.1 --port 8080

python main.py train-segmentation --dataset brats --model-backend monai_unet [SEGMENTATION_TRAIN_ARGS]
python main.py train-segmentation-h5 --data-dir res/datasets/archive/BraTS2020_training_data [H5_TRAIN_ARGS]
python main.py train-classifier --dataset brats [CLASSIFIER_TRAIN_ARGS]
python main.py infer \
  --dataset brats \
  --sample-path /path/to/sample_or_patient_dir \
  --segmentation-checkpoint res/models/best_segmentation_model.pth \
  [--classifier-checkpoint res/models/tumor_classifier.pth] \
  [--model-backend monai_unet] \
  [--classifier-threshold 0.5] \
  [--save-prediction-path res/predictions/patient_pred.nii.gz]
```

**Using uv:**
```bash
uv run main.py datasets
uv run main.py datasets --search-dir /path/to/data --max-depth 5
uv run main.py models
uv run main.py web --host 127.0.0.1 --port 8080

uv run main.py train-segmentation --dataset brats --model-backend monai_unet [SEGMENTATION_TRAIN_ARGS]
uv run main.py train-segmentation-h5 --data-dir res/datasets/archive/BraTS2020_training_data [H5_TRAIN_ARGS]
uv run main.py train-classifier --dataset brats [CLASSIFIER_TRAIN_ARGS]
uv run main.py infer \
  --dataset brats \
  --sample-path /path/to/sample_or_patient_dir \
  --segmentation-checkpoint res/models/best_segmentation_model.pth \
  [--classifier-checkpoint res/models/tumor_classifier.pth] \
  [--model-backend monai_unet] \
  [--classifier-threshold 0.5] \
  [--save-prediction-path res/predictions/patient_pred.nii.gz]
```

`datasets` helps you find known datasets and register the dataset type.
`models` shows available segmentation backends (currently `monai_unet` and `nnunet` placeholder).
`web` starts a barebones browser interface for inference and preview visualization.

For W&B logging, place `WANDB_API_KEY` (and optionally `WANDB_ENTITY`) in `.env`. Training commands load `.env` automatically.

To add a new dataset format:
1. Create a new adapter in `src/data/adapters/` implementing `DatasetAdapter`.
2. Register it in `src/data/registry.py`.
3. Use `python main.py datasets --search-dir /path/to/data` to confirm discovery.
4. Train/infer with `--dataset <your-adapter-name>`.

To add a new segmentation training backend:
1. Implement a backend in `src/segmentation/backends/` by extending `SegmentationBackend`.
2. Register it in `src/segmentation/registry.py`.
3. Train with `python main.py train-segmentation --model-backend <backend-name>`.

`infer` runs a clean two-step pipeline:
1. Classifier checks if tumor is present.
2. Segmentation runs only when classifier probability is above threshold.

Default preprocessing (train and inference):
1. Replace NaN/Inf with finite values.
2. Clip each channel to non-zero 0.5/99.5 percentiles.
3. Apply per-channel non-zero z-score normalization.

In the web UI, after each run you get:
1. A browser-rendered preview image (input slice + segmentation overlay).
2. A downloadable prediction mask file in dataset-native format.

<!-- TODO: Instructions on how to run the project and use its features. -->

### Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/PROJECT-TEMPLATE/](https://cogitontnu.github.io/PROJECT-TEMPLATE/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.

<table align="center">
    <tr>
        <!--
        <td align="center">
            <a href="https://github.com/NAME_OF_MEMBER">
              <img src="https://github.com/NAME_OF_MEMBER.png?size=100" width="100px;" alt="NAME OF MEMBER"/><br />
              <sub><b>NAME OF MEMBER</b></sub>
            </a>
        </td>
        -->
    </tr>
</table>

![Group picture](docs/img/team.png)

### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
