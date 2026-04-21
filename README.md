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
- **Python 3.13**: Required for the project. [Download Python](https://www.python.org/downloads/)
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

The project solves two tasks on BraTS 2020 MRI volumes:

1. **Segmentation** (primary): a 3D network predicts the BraTS tumor
   sub-regions (TC / WT / ET) from four co-registered MRI modalities
   (FLAIR, T1, T1ce, T2).
2. **Classification** (secondary): a rule-based stage turns each predicted
   mask into a binary `tumor` / `no_tumor` label plus a coarse phenotype.

Unpack the BraTS training release under
`res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/` (the default `--data-dir`).

### Full pipeline on a vast.ai (or any CUDA) instance

One command from a fresh GPU host:

```bash
git clone <repo> cancervision && cd cancervision
cp .env.example .env   # put WANDB_API_KEY, optional DATA_DIR / DATA_ARCHIVE
bash scripts/run_vast.sh   # extra flags forward to training, e.g. --max-epochs 50
```

`scripts/run_vast.sh` installs `uv`, syncs the env, fetches the dataset if
`DATA_ARCHIVE` points to a local/remote `.tar.gz` / `.tar` / `.zip`, then calls
`python -m src.pipeline` which runs preflight (CUDA, BraTS tree integrity, W&B
credentials) before launching training and logging to Weights & Biases.

### Train a segmentation model

Training is a single CLI; the model is picked from the registry in
`src.models.registry` via `--model`.

```bash
uv run python -m src.training.train --model dynunet
```

Only `dynunet` is wired end-to-end today. A reference `unet` builder is
registered so the multi-model plumbing is exercised. To add a new model,
implement a builder and register it in `src/models/registry.py`; see
`docs/manuals/ml-lifecycle-setup.md` for the full recipe.

Full flag list: `python -m src.training.train --help`.

### Run segmentation inference

```bash
uv run python -m src.inference.inference \
  --model-id dynunet_latest \
  --input-root /path/to/MICCAI_BraTS2020_TrainingData \
  --output-root res/predictions/dynunet_latest
```

Deployable checkpoints are declared in `res/models/model_registry.json`.

### Classify predicted segmentations

Binary cancer-vs-non-cancer plus phenotype, derived from predicted masks:

```bash
uv run python -m src.classification.classify \
  --input-root res/predictions/dynunet_latest \
  --output-csv res/classification/reports/dynunet_latest.csv
```

A case is labelled `tumor` when the predicted Whole-Tumor region has at least
`--min-tumor-voxels` voxels (default 16), otherwise `no_tumor`.

### Web interface (drag-and-drop inference)

Serve a browser-based inference UI that accepts arbitrary model weights and
the four BraTS MRI modalities via drag-and-drop:

```bash
uv run python -m src.web --host 127.0.0.1 --port 8080
```

See `docs/manuals/web-interface.md` for details.

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
