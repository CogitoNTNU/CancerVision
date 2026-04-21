# Dataset
**We purpously omitted including the dataset in our repository to follow the republishing constraints specified under the dataset's license agreement.**

The HNTSMRG24 dataset can be downloaded from **https://zenodo.org/records/11199559**. After download, extract the training data into `res/HNTSMRG24_train/`.

# Models and Inference Outputs

- Store inference-ready checkpoints in `res/models/`.
- Register deployable models in `res/models/model_registry.json`.
- Generated segmentation outputs are written to `res/predictions/` by default and are gitignored.

# Training and Classification Configs

- Store training experiment definitions in `res/configs/training_registry.json`.
- Store classifier definitions in `res/classification/classifier_registry.json`.
- Save classifier artifacts in `res/classification/models/` and reports in `res/classification/reports/`.