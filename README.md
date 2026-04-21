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

## Datasets

This project leverages several brain imaging datasets for model training and validation:

- Juvekar, P., Dorent, R., Kögl, F., Torio, E., Barr, C., Rigolo, L., Galvin, C., Jowkar, N., Kazi, A., Haouchine, N., Cheema, H., Navab, N., Pieper, S., Wells, W. M., Bi, W. L., Golby, A., Frisken, S., & Kapur, T. (2023). The Brain Resection Multimodal Imaging Database (ReMIND) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/3RAG-D070

- Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37

- Moreau, N. N., Leclercq, A. G., Desmonts, A., Poirier, Y., Dubru, A., Guillemette, L., Lecoeur, P., Lemasson, K., Jaudet, C., Brunaud, C., Valable, S., Geffrelot, J., Stefan, D., Leleu, T., Raboutet, C., Le Henaff, L., Batalla, A., Lacroix, J., Rouzier, R., & Corroyer-Dulmont, A. (2025). Pre and post treatment MRI and radiotherapy plans of patients with glioblastoma: the CFB-GBM cohort (CFB-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/V9PN-2F72

- Gagnon, L., Gupta, D., Mastorakos, G., White, N., Goodwill, V., McDonald, C., Beaumont, T., Conlin, C., Seibert, T., Nguyen, U., Hattangadi-Gluth, J., Kesari, S., Schulte, J., Piccioni, D., Schmainda, K., Farid, N., Dale, A., Rudie, J. (2025). The University of California San Diego annotated post-treatment high-grade glioma multimodal MRI dataset (UCSD-PTGBM) (Version 3) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/fwv2-dt74

- Reddy, D., Saadat, N., Holcomb, J., Wagner, B., Truong, N., Bowerman, J., Hatanpaa, K., Patel, T., Pinho, M., Yu, F., Zhang, K., Lodhi, S., Madhuranthakam, A., Bangalore Yogananda, C. G., & Maldjian, J. (2026). The University of Texas Southwestern Glioma MRI dataset with molecular marker characterization and segmentations (UTSW-Glioma) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/DFAE-1B86

- Chadha, S., Weiss, D., Janas, A., Ramakrishnan, D., Hager, T., Osenberg, K., Willms, K., Zhu, J., Chiang, V., Bakas, S., Maleki, N., Sritharan, D. V., Schoenherr, S., Westerhoff, M., Zawalich, M., Davis, M., Malhotra, A., Bousabarah, K., Deusch, C., Lin, M., Aneja, S., & Aboian, M. S. (2025). Yale longitudinal dataset of brain metastases on MRI with associated clinical data (Yale-Brain-Mets-Longitudinal) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/3YAT-E768

- Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos, C. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49

- Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm (version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73

## Usage

Local CancerVision training from repo root:

```bash
uv run python -m src.models.dynnet \
  --dataset-source cancervision_binary_seg \
  --task-manifest res/dataset/cancervision-standardized/task_manifests/segmentation_binary_curated.csv
```

Canonical SLURM launch:

```bash
sbatch --account=<account> --constraint=gpu40g scripts/train_dynnet.sbatch
# or
sbatch --account=<account> --constraint=gpu80g scripts/train_dynnet.sbatch
```

Trainer now uses one batch script and auto-detects GPU profile from Slurm constraints. Old split scripts like `clean_slurm*.slurm` and `run_slurm.slurm` are removed. To run raw BraTS folders instead of CancerVision manifests, submit with `DATASET_SOURCE=brats` and `DATA_DIR=/path/to/brats/root`.

If one manifest must work across local and cluster path layouts, remap prefixes at runtime:

```bash
uv run python -m src.models.dynnet \
  --dataset-source cancervision_binary_seg \
  --task-manifest res/dataset/cancervision-standardized/task_manifests/segmentation_binary_broad.csv \
  --path-prefix-map 'Z:\dataset\cancervision-standardized=/cluster/home/eldarja/CancerVision/res/dataset/cancervision-standardized'
```

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
